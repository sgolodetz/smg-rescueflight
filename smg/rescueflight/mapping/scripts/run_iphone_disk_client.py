import cv2
import json
import numpy as np
import os

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import CameraParameters, GeometryUtil, ImageUtil, PooledQueue


def try_load_frame(frame_idx: int, sequence_dir: str) -> Optional[Dict[str, Any]]:
    """
    Try to load a frame from an iPhone RGB-D sequence.

    .. note::
        The iPhone app that produces the sequences we consume is "3d Scanner App", by Laan Labs.

    :param frame_idx:       The frame index.
    :param sequence_dir:    The sequence directory.
    :return:                The RGB-D frame, if possible, or None otherwise.
    """
    # Determine the names of the files.
    colour_filename: str = os.path.join(sequence_dir, f"frame_{frame_idx:05d}.jpg")
    conf_filename: str = os.path.join(sequence_dir, f"conf_{frame_idx:05d}.png")
    depth_filename: str = os.path.join(sequence_dir, f"depth_{frame_idx:05d}.png")
    json_filename: str = os.path.join(sequence_dir, f"frame_{frame_idx:05d}.json")

    # If one of the essential files doesn't exist, early out.
    if not os.path.exists(conf_filename) or not os.path.exists(depth_filename) or not os.path.exists(json_filename):
        return None

    # Load whatever components of the frame exist.
    colour_image: Optional[np.ndarray] = None
    if os.path.exists(colour_filename):
        colour_image = cv2.imread(colour_filename)

    conf_image: np.ndarray = cv2.imread(conf_filename, cv2.IMREAD_UNCHANGED)
    depth_image: np.ndarray = ImageUtil.from_short_depth(cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED))

    with open(json_filename) as f:
        data: Dict[str, Any] = json.load(f)

    world_from_camera: np.ndarray = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    world_from_camera = np.linalg.inv(CameraPoseConverter.modelview_to_pose(np.linalg.inv(world_from_camera)))

    # Rotate the pose by 180 degrees about the x axis (so that the model will be the right way up, with the y axis
    # pointing downwards as per our convention).
    m: np.ndarray = np.eye(4)
    m[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * np.pi).as_matrix()
    world_from_camera = m @ world_from_camera

    # Filter out low-confidence pixels from the depth image.
    depth_image[conf_image < 2] = 0.0

    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        "world_from_camera": world_from_camera
    }


def try_load_obb(sequence_dir: str) -> Optional[np.ndarray]:
    """
    Try to load in the oriented bounding box for an iPhone RGB-D sequence.

    :param sequence_dir:    The sequence directory.
    :return:                The oriented bounding box for the sequence, if possible, or None otherwise.
    """
    # Note: This is currently unused, but may ultimately be useful for aligning the model, so I'm keeping it for now.
    json_filename: str = os.path.join(sequence_dir, "info.json")

    try:
        with open(json_filename) as f:
            data: Dict[str, Any] = json.load(f)
            return np.array(data["userOBB"]["points"])
    except FileNotFoundError:
        return None


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
            # FIXME: These should ultimately be loaded in rather than hard-coded.
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
            initial_pos: Optional[np.ndarray] = None
            pause: bool = not batch

            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir)

                # If the frame was successfully loaded:
                if frame is not None:
                    # Get the frame's pose.
                    world_from_camera: np.ndarray = frame["world_from_camera"]

                    # If the initial camera position hasn't yet been recorded, record it now.
                    if initial_pos is None:
                        initial_pos = world_from_camera[0:3, 3].copy()

                    # Subtract the initial camera position from the current camera position (this has the effect
                    # of centring the reconstruction on the initial camera position).
                    world_from_camera[0:3, 3] -= initial_pos

                    m = np.array([
                        [0.809, 0, 0.5878, 0],
                        [0, 1, 0, -1.4],
                        [-0.5878, 0, 0.809, 0.1],
                        [0, 0, 0, 1]
                    ])
                    world_from_camera = m @ world_from_camera

                    # If this frame has a valid colour image:
                    if frame["colour_image"] is not None:
                        # Make a resized version of the colour image that is the same size as the depth image.
                        colour_image = cv2.resize(frame["colour_image"], (256, 192))

                        # Make a note of the depth image so that it can be shown later.
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
