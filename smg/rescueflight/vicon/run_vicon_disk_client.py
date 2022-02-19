import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.utility import CameraParameters, ImageUtil, PooledQueue, PoseUtil, SequenceUtil
from smg.vicon import OfflineViconInterface, SubjectFromSourceCache


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
    parser.add_argument(
        "--source_subject", type=str, default="Tello",
        help="the name of the Vicon subject corresponding to the source"
    )
    parser.add_argument(
        "--use_tracked_poses", action="store_true",
        help="whether to use the tracked camera poses stored with the sequence instead of the Vicon-based poses"
    )
    parser.add_argument(
        "--use_vicon_scale", action="store_true",
        help="whether to convert the tracked poses to Vicon scale using a pre-calculated scale factor"
    )
    args: dict = vars(parser.parse_args())

    batch: bool = args["batch"]
    sequence_dir: str = args["sequence_dir"]
    source_subject: str = args["source_subject"]
    use_tracked_poses: bool = args["use_tracked_poses"]
    use_vicon_scale: bool = args["use_vicon_scale"]

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(sequence_dir)

    # Connect to the Vicon interface.
    with OfflineViconInterface(folder=sequence_dir) as vicon:
        # Connect to the mapping server.
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_WAIT
        ) as client:
            # Try to load the camera parameters for the sequence. If this fails, raise an exception.
            calib: Optional[CameraParameters] = SequenceUtil.try_load_calibration(sequence_dir)
            if calib is None:
                raise RuntimeError(f"Cannot load calibration from '{sequence_dir}'")

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            # Prepare the variables needed to process the sequence.
            colour_image: Optional[np.ndarray] = None
            initial_from_vicon: Optional[np.ndarray] = None
            pause: bool = not batch
            shown_colour_image: bool = False

            # If we're using tracked poses and they're to be converted to Vicon scale, load in the scale factor.
            scale_factor: float = 1.0
            if use_tracked_poses and use_vicon_scale:
                scale_filename: str = os.path.join(sequence_dir, "reconstruction", "vicon_from_world_scale_factor.txt")
                with open(scale_filename, "r") as file:
                    scale_factor = float(file.readline())

            # Until the user wants to quit:
            while True:
                # Try to load a Vicon frame from disk. If that succeeds:
                if vicon.get_frame():
                    # Get the associated frame number.
                    frame_number: int = vicon.get_frame_number()

                    # Try to load in the associated colour image (originally captured by the image source).
                    colour_image_filename: str = os.path.join(sequence_dir, f"{frame_number}.color.png")
                    colour_image = cv2.imread(colour_image_filename)

                    # If that succeeds:
                    if colour_image is not None:
                        # Try to determine the camera pose.
                        pose: Optional[np.ndarray] = None

                        if use_tracked_poses:
                            pose_filename: str = os.path.join(sequence_dir, f"{frame_number}.pose.txt")
                            pose = PoseUtil.load_pose(pose_filename)
                            if use_vicon_scale:
                                pose[0:3, 3] *= scale_factor
                        else:
                            vicon_from_source: Optional[np.ndarray] = vicon.get_image_source_pose(
                                source_subject, subject_from_source_cache
                            )

                            if vicon_from_source is not None:
                                if initial_from_vicon is None:
                                    initial_from_vicon = np.linalg.inv(vicon_from_source)

                                pose = initial_from_vicon @ vicon_from_source

                        # If the camera pose is available:
                        if pose is not None:
                            # Construct a dummy depth image.
                            dummy_depth_image: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.float32)

                            # Send the frame across to the server.
                            client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                                frame_number, colour_image, ImageUtil.to_short_depth(dummy_depth_image), pose, msg
                            ))
                        else:
                            print(f"Warning: Missing pose for frame {frame_number}")
                    else:
                        print(f"Warning: Missing colour image for frame {frame_number}")

                # Otherwise, if we're in batch mode, exit.
                elif batch:
                    # noinspection PyProtectedMember
                    os._exit(0)

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("Vicon Disk Client", colour_image)
                    shown_colour_image = True

                # If a colour image has ever been shown:
                if shown_colour_image:
                    # Wait for a keypress, processing OpenCV events in the background as needed.
                    if pause:
                        c = cv2.waitKey()
                    else:
                        c = cv2.waitKey(50)

                    # Allow the user to pause/unpause the playback, and exit if desired.
                    if c == ord('b'):
                        pause = False
                    elif c == ord('n'):
                        pause = True
                    elif c == ord('q'):
                        cv2.destroyAllWindows()
                        break


if __name__ == "__main__":
    main()
