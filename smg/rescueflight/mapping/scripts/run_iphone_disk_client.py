import cv2
import json
import numpy as np
import os

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.rigging.cameras import SimpleCamera
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
    world_from_camera = np.linalg.inv(CameraPoseConverter.modelview_to_pose(np.linalg.inv(world_from_camera)))

    m = np.eye(4)
    m[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * np.pi).as_matrix()
    world_from_camera = m @ world_from_camera

    # TODO
    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        "world_from_camera": world_from_camera
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
            # initial_cam: Optional[SimpleCamera] = None
            # initial_from_world: Optional[np.ndarray] = None
            initial_pos: Optional[np.ndarray] = None
            pause: bool = not batch

            while True:
                # TODO
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir)

                # If the frame was successfully loaded.
                if frame is not None:
                    # TODO
                    world_from_camera: np.ndarray = frame["world_from_camera"]

                    # TODO
                    # if initial_from_world is None:
                    #     initial_from_world = np.linalg.inv(world_from_camera)
                    # initial_from_camera: np.ndarray = initial_from_world @ world_from_camera

                    # TODO
                    if initial_pos is None:
                        initial_pos = world_from_camera[0:3, 3].copy()
                    # if initial_cam is None:
                    #     initial_cam = CameraPoseConverter.pose_to_camera(np.linalg.inv(world_from_camera))
                    #     print(initial_cam.p(), initial_cam.n(), initial_cam.u(), initial_cam.v())

                    # TODO
                    # print(world_from_camera)
                    world_from_camera[0:3, 3] -= initial_pos
                    # print(world_from_camera)

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
                            world_from_camera,
                            msg
                        ))

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
                        c = cv2.waitKey(1)

                    if c == ord('b'):
                        pause = False
                    elif c == ord('n'):
                        pause = True
                    elif c == ord('q'):
                        # noinspection PyProtectedMember
                        os._exit(0)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
