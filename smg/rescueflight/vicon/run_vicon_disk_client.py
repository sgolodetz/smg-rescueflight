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
    args: dict = vars(parser.parse_args())

    sequence_dir: str = args["sequence_dir"]
    source_subject: str = args["source_subject"]
    use_tracked_poses: bool = args["use_tracked_poses"]

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

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
            frame_idx: int = 0
            initial_from_world: Optional[np.ndarray] = None
            pause: bool = True

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
                        else:
                            subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(
                                source_subject
                            )
                            subject_from_world: Optional[np.ndarray] = vicon.get_segment_global_pose(
                                source_subject, source_subject
                            )

                            if subject_from_source is not None and subject_from_world is not None:
                                world_from_subject: np.ndarray = np.linalg.inv(subject_from_world)
                                world_from_source: np.ndarray = world_from_subject @ subject_from_source
                                if initial_from_world is None:
                                    initial_from_world = np.linalg.inv(world_from_source)

                                pose = initial_from_world @ world_from_source

                        # If the camera pose is available:
                        if pose is not None:
                            # Construct a dummy depth image.
                            dummy_depth_image: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.float32)

                            # Send the frame across to the server.
                            client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                                frame_idx, colour_image, ImageUtil.to_short_depth(dummy_depth_image), pose, msg
                            ))
                        else:
                            print(f"Warning: Missing pose for frame {frame_idx}")
                    else:
                        print(f"Warning: Missing colour image for frame {frame_idx}")

                    # Increment the frame index.
                    frame_idx += 1

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("Vicon Disk Client", colour_image)

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
