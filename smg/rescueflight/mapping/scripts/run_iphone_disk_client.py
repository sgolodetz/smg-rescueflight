import cv2
import json
import numpy as np
import os

from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import CameraParameters, GeometryUtil, ImageUtil, PooledQueue


def try_load_frame(frame_idx: int, sequence_dir: str) -> Optional[Dict[str, Any]]:
    """
    TODO

    :param frame_idx:       TODO
    :param sequence_dir:    TODO
    :return:                TODO
    """
    # Determine the names of the files.
    colour_filename: str = os.path.join(sequence_dir, f"frame_{frame_idx:05d}.jpg")
    depth_filename: str = os.path.join(sequence_dir, f"depth_{frame_idx:05d}.png")
    json_filename: str = os.path.join(sequence_dir, f"frame_{frame_idx:05d}.json")

    # If one of the essential files doesn't exist, early out.
    if not os.path.exists(depth_filename) or not os.path.exists(json_filename):
        return None

    # Load whatever components of the frame exist.
    colour_image: Optional[np.ndarray] = None
    if os.path.exists(colour_filename):
        colour_image = cv2.imread(colour_filename)

    depth_image: np.ndarray = ImageUtil.from_short_depth(cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED))

    with open(json_filename) as f:
        data: Dict[str, Any] = json.load(f)

    world_from_camera: np.ndarray = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    # world_from_camera = np.array([
    #     [-1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ]) @ world_from_camera

    # projection_matrix: np.ndarray = np.array(data["projectionMatrix"]).reshape(4, 4)
    transform_to_world_map: np.ndarray = np.array([
        [0.99452191591262817, 0, -0.10452836751937866, -1.9408578872680664],
        [0, 1, 0, -0.97300004959106445],
        [0.10452836751937866, 0, 0.99452191591262817, -0.41681873798370361],
        [0, 0, 0, 1]
    ])

    # world_from_camera: np.ndarray = np.eye(4)
    # world_from_camera[0:3, 3] = np.array([0, 0, -5])
    world_from_camera = np.linalg.inv(CameraPoseConverter.modelview_to_pose(np.linalg.inv(world_from_camera)))
    from scipy.spatial.transform import Rotation as R
    m = np.eye(4)
    m[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * np.pi).as_matrix()
    world_from_camera = m @ world_from_camera

    # TODO
    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        # "projection_matrix": projection_matrix,
        "world_from_camera": world_from_camera  # np.eye(4)  # transform_to_world_map  #  @ np.linalg.inv(world_from_camera)
    }


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    batch: bool = args["batch"]
    sequence_dir: str = args["sequence_dir"]

    # FIXME: Delete.
    from typing import List
    trajectory: List[Tuple[float, np.ndarray]] = []

    try:
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_WAIT
        ) as client:
            # Send a calibration message to tell the server the camera parameters.
            calib: CameraParameters = CameraParameters()
            colour_intrinsics: Tuple[float, float, float, float] = (
                1439.68017578125, 1439.68017578125, 961.381103515625, 727.34173583984375
            )
            intrinsics: Tuple[float, float, float, float] = GeometryUtil.rescale_intrinsics(
                colour_intrinsics, (1920, 1440), (256, 192)
            )
            calib.set("colour", 256, 192, *intrinsics)
            calib.set("depth", 256, 192, *intrinsics)

            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            # Initialise some variables.
            colour_image: Optional[np.ndarray] = None
            depth_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            pause: bool = not batch

            while True:
                # TODO
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir)

                # If the frame was successfully loaded.
                if frame is not None:
                    # TODO
                    if frame["colour_image"] is not None:
                        # TODO
                        colour_image = cv2.resize(frame["colour_image"], (256, 192))
                        depth_image = frame["depth_image"]

                        # Send the frame across to the server.
                        client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                            frame_idx,
                            colour_image,
                            ImageUtil.to_short_depth(depth_image),
                            frame["world_from_camera"],
                            msg
                        ))

                        # FIXME: Delete.
                        trajectory.append((frame_idx, frame["world_from_camera"]))

                    # Increment the frame index.
                    frame_idx += 1

                # Otherwise, if we're in batch mode, exit.
                elif batch:
                    # noinspection PyProtectedMember
                    os._exit(0)

                # Show the most recent colour and depth images (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("Colour Image", colour_image)
                    cv2.imshow("Depth Image", depth_image / 5)

                    if pause and frame is not None and frame["colour_image"] is not None:
                        c = cv2.waitKey()
                    else:
                        c = cv2.waitKey(1)  # 50)

                    if c == ord('b'):
                        pause = False
                    elif c == ord('n'):
                        pause = True
                    elif c == ord('q'):
                        # FIXME: Delete.
                        # import open3d as o3d
                        # from smg.open3d import VisualisationUtil
                        # grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid(
                        #     [-2, -2, -2], [2, 0, 2], [1, 1, 1]
                        # )
                        # segments: o3d.geometry.LineSet = VisualisationUtil.make_trajectory_segments(
                        #     trajectory, colour=(0.0, 1.0, 0.0)
                        # )
                        #
                        # # Visualise the geometries.
                        # VisualisationUtil.visualise_geometries([segments])

                        # noinspection PyProtectedMember
                        os._exit(0)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
