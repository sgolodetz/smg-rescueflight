import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Any, Dict, Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.pyorbslam2 import RGBDTracker
from smg.utility import CameraParameters, ImageUtil, PooledQueue, PoseUtil


def try_load_frame(frame_idx: int, sequence_dir: str) -> Optional[Dict[str, Any]]:
    """
    Try to load a frame from a ScanNet sequence.

    .. note::
        The RGB-D frame is returned as a mapping from strings to pieces of data:
            "colour_image" -> np.ndarray (an 8UC3 image)
            "depth_image" -> np.ndarray (a float image)
            "world_from_camera" -> np.ndarray (a 4x4 transformation matrix)

    :param frame_idx:       The frame index.
    :param sequence_dir:    The sequence directory.
    :return:                The RGB-D frame, if possible, or None otherwise.
    """
    # Determine the names of the colour image,  depth image and pose files.
    colour_filename: str = os.path.join(sequence_dir, "color", f"{frame_idx}.jpg")
    depth_filename: str = os.path.join(sequence_dir, "depth", f"{frame_idx}.png")
    pose_filename: str = os.path.join(sequence_dir, "pose", f"{frame_idx}.txt")

    # If one of the files doesn't exist, early out.
    if not os.path.exists(colour_filename) or not os.path.exists(depth_filename) or not os.path.exists(pose_filename):
        return None

    # Otherwise, load and return the frame.
    colour_image: np.ndarray = cv2.imread(colour_filename)
    depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename, depth_scale_factor=1000)
    world_from_camera: np.ndarray = PoseUtil.load_pose(pose_filename)

    height, width = colour_image.shape[:2]
    depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        "world_from_camera": world_from_camera
    }


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--canonicalise_poses", action="store_true",
        help="whether to canonicalise the poses (i.e. start the camera trajectory from the identity)"
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
    canonicalise_poses: bool = args["canonicalise_poses"]
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
            # Hard-code the camera parameters (for now). Note that we upsample the depth images to be the same
            # size as the colour images, since that's what Open3D expects, so we need to change the intrinsics
            # here to reflect that as well.
            calib: CameraParameters = CameraParameters()
            calib.set("colour", 1296, 968, 1169.621094, 1167.105103, 646.295044, 489.927032)
            calib.set("depth", 1296, 968, 1169.621094, 1167.105103, 646.295044, 489.927032)
            # calib.set("depth", 640, 480, 577.590698, 578.729797, 318.905426, 242.683609)

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            # Initialise some variables.
            colour_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            initial_from_world: Optional[np.ndarray] = None
            pause: bool = not batch_mode

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir)

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

                    # Canonicalise the camera pose if requested.
                    if canonicalise_poses:
                        if initial_from_world is None:
                            initial_from_world = np.linalg.inv(frame["world_from_camera"])

                        frame["world_from_camera"] = initial_from_world @ frame["world_from_camera"]

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
                    cv2.imshow("ScanNet Client", colour_image)

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
