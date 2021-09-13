import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.pyorbslam2 import RGBDTracker
from smg.utility import CameraParameters, ImageUtil, PooledQueue, TrajectoryUtil


def try_load_frame(frame_idx: int, sequence_dir: str, trajectory: List[Tuple[float, np.ndarray]]) \
        -> Optional[Dict[str, Any]]:
    """
    Try to load a frame from an ICL-NUIM sequence.

    .. note::
        The RGB-D frame is returned as a mapping from strings to pieces of data:
            "colour_image" -> np.ndarray (an 8UC3 image)
            "depth_image" -> np.ndarray (a float image)
            "world_from_camera" -> np.ndarray (a 4x4 transformation matrix)

    :param frame_idx:       The frame index.
    :param sequence_dir:    The sequence directory.
    :param trajectory:      The camera trajectory.
    :return:                The RGB-D frame, if possible, or None otherwise.
    """
    # TODO
    if frame_idx >= len(trajectory):
        return None

    # Determine the names of the colour image and depth image files.
    colour_filename: str = os.path.join(sequence_dir, "rgb", f"{frame_idx}.png")
    depth_filename: str = os.path.join(sequence_dir, "depth", f"{frame_idx}.png")

    # If one of the files doesn't exist, early out.
    if not os.path.exists(colour_filename) or not os.path.exists(depth_filename):
        return None

    # Otherwise, load and return the frame.
    return {
        "colour_image": cv2.imread(colour_filename),
        "depth_image": ImageUtil.load_depth_image(depth_filename, depth_scale_factor=5000),
        "world_from_camera": trajectory[frame_idx][1].copy()
    }


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    parser.add_argument(
        "--use_tracker", action="store_true", help="whether to use a tracker instead of the ground-truth trajectory"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: str = args["sequence_dir"]
    use_tracker: bool = args["use_tracker"]

    tracker: Optional[RGBDTracker] = None
    if use_tracker:
        tracker = RGBDTracker(
            settings_file="settings-icl_nuim.yaml", use_viewer=True, voc_file="C:/orbslam2/Vocabulary/ORBvoc.bin",
            wait_till_ready=True
        )

    try:
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_WAIT
        ) as client:
            # Hard-code the camera parameters as per https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats.
            calib: CameraParameters = CameraParameters()
            calib.set("colour", 640, 480, 525.0, 525.0, 319.5, 239.5)
            calib.set("depth", 640, 480, 525.0, 525.0, 319.5, 239.5)

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            # Load the camera trajectory.
            trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
                os.path.join(sequence_dir, "trajectory.txt")
            )

            colour_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            pause: bool = True

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir, trajectory)

                # If the frame was successfully loaded:
                if frame is not None:
                    # TODO
                    if use_tracker:
                        # TODO
                        camera_from_world: Optional[np.ndarray] = tracker.estimate_pose(
                            frame["colour_image"], frame["depth_image"]
                        )

                        # TODO
                        if camera_from_world is not None:
                            frame["world_from_camera"] = np.linalg.inv(camera_from_world)

                    # Send it across to the server.
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

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("ICL-NUIM Client", colour_image)

                    if pause:
                        c = cv2.waitKey()
                    else:
                        c = cv2.waitKey(50)

                    if c == ord('b'):
                        pause = False
                    elif c == ord('n'):
                        pause = True
                    elif c == ord('q'):
                        break
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
