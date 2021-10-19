import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.pyorbslam2 import RGBDTracker
from smg.utility import CameraParameters, ImageUtil, PooledQueue, PoseUtil


def try_load_frame(frame_idx: int, sequence_dir: str, frame_numbers: List[int]) -> Optional[Dict[str, Any]]:
    """
    Try to load a frame from a GTA-IM sequence.

    .. note::
        The RGB-D frame is returned as a mapping from strings to pieces of data:
            "colour_image" -> np.ndarray (an 8UC3 image)
            "depth_image" -> np.ndarray (a float image)
            "world_from_camera" -> np.ndarray (a 4x4 transformation matrix)

    :param frame_idx:       The frame index.
    :param sequence_dir:    The sequence directory.
    :param frame_numbers:   TODO
    :return:                The RGB-D frame, if possible, or None otherwise.
    """
    # TODO
    if frame_idx >= len(frame_numbers):
        return None

    # Determine the names of the colour image,  depth image and pose files.
    colour_filename: str = os.path.join(sequence_dir, f"{frame_numbers[frame_idx]:05d}.jpg")
    depth_filename: str = os.path.join(sequence_dir, f"{frame_numbers[frame_idx]:05d}.png")

    # If one of the files doesn't exist, early out.
    if not os.path.exists(colour_filename) or not os.path.exists(depth_filename):
        return None

    # Otherwise, load and return the frame.
    colour_image: np.ndarray = cv2.imread(colour_filename)
    # depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename, depth_scale_factor=1000)
    depth_image: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.uint8)
    world_from_camera: np.ndarray = np.eye(4)

    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        "world_from_camera": world_from_camera
    }


def get_frame_number(filename: str) -> int:
    """
    Get the frame number corresponding to a GTA-IM image file.

    .. note::
        The files are named <frame number>.jpg, so we can get the frame numbers directly from the file names.

    :param filename:    The name of a GTA-IM image file.
    :return:            The corresponding frame number.
    """
    frame_number, _ = filename.split(".")
    return int(frame_number)


def main() -> None:
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
    parser.add_argument(
        "--use_tracker", action="store_true",
        help="whether to use the tracker instead of the ground-truth trajectory"
    )
    args: dict = vars(parser.parse_args())

    batch_mode: bool = args["batch"]
    sequence_dir: str = args["sequence_dir"]
    use_tracker: bool = args["use_tracker"]

    # Construct the camera tracker if needed.
    tracker: Optional[RGBDTracker] = None
    if use_tracker:
        tracker = RGBDTracker(
            settings_file="settings-scannet.yaml", use_viewer=True, voc_file="C:/orbslam2/Vocabulary/ORBvoc.bin",
            wait_till_ready=True
        )

    try:
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_WAIT
        ) as client:
            # Hard-code the camera parameters (for now).
            # FIXME: See https://github.com/ZheC/GTA-IM-Dataset for details on how to obtain the parameters properly.
            calib: CameraParameters = CameraParameters()
            calib.set("colour", 1920, 1080, 1158.03373708, 1158.03373708, 960.0, 540.0)
            calib.set("depth", 1920, 1080, 1158.03373708, 1158.03373708, 960.0, 540.0)

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            frame_numbers: List[int] = sorted(
                [get_frame_number(f) for f in os.listdir(sequence_dir) if f.endswith(".jpg")],
            )

            # Initialise some variables.
            colour_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            pause: bool = not batch_mode

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir, frame_numbers)

                # If the frame was successfully loaded:
                if frame is not None:
                    # If we're using the camera tracker:
                    if use_tracker:
                        # Try to estimate the camera pose.
                        camera_from_world: Optional[np.ndarray] = tracker.estimate_pose(
                            frame["colour_image"], frame["depth_image"]
                        )

                        # If that succeeds, replace the ground-truth pose in the frame with the tracked pose.
                        if camera_from_world is not None:
                            frame["world_from_camera"] = np.linalg.inv(camera_from_world)

                    # Send the frame across to the server.
                    client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                        frame_idx,
                        frame["colour_image"],
                        ImageUtil.to_short_depth(frame["depth_image"]),
                        frame["world_from_camera"],
                        msg
                    ))

                    # Increment the frame index.
                    frame_idx += 1

                    # Update the colour image so that it can be shown.
                    colour_image = frame["colour_image"]

                # Otherwise, if we're in batch mode, exit.
                elif batch_mode:
                    # noinspection PyProtectedMember
                    os._exit(0)

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("GTA-IM Client", colour_image)

                    if pause:
                        c = cv2.waitKey()
                    else:
                        c = cv2.waitKey(50)

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
