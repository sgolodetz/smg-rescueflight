import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.utility import CameraParameters, ImageUtil, PooledQueue, RGBDSequenceUtil
from smg.vicon import OfflineViconInterface, SubjectFromSourceCache


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--image_source_subject", type=str, default="Tello",
        help="the name of the Vicon subject from which the images were captured"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: str = args["sequence_dir"]

    from smg.pyorbslam2 import MonocularTracker
    tracker: MonocularTracker = MonocularTracker(
        settings_file=f"settings-tello.yaml", use_viewer=True,
        voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=True
    )

    try:
        with OfflineViconInterface(folder=sequence_dir) as vicon:
            with MappingClient(
                frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
                pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
            ) as client:
                # TODO
                subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

                # Try to load the camera parameters for the sequence. If this fails, raise an exception.
                # FIXME: Move try_load_calibration somewhere more central.
                calib: Optional[CameraParameters] = RGBDSequenceUtil.try_load_calibration(sequence_dir)
                if calib is None:
                    raise RuntimeError(f"Cannot load calibration from '{sequence_dir}'")

                # Send a calibration message to tell the server the camera parameters.
                client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                    calib.get_image_size("colour"), calib.get_image_size("depth"),
                    calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
                ))

                colour_image: Optional[np.ndarray] = None
                world_from_initial: Optional[np.ndarray] = None
                frame_idx: int = 0
                pause: bool = True

                # Until the user wants to quit:
                while True:
                    # Try to load a Vicon frame from disk. If that succeeds:
                    if vicon.get_frame():
                        frame_number: int = vicon.get_frame_number()
                        colour_image_filename: str = os.path.join(sequence_dir, f"{frame_number}.png")
                        colour_image = cv2.imread(colour_image_filename)
                        if colour_image is not None:
                            dummy_depth_image: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.float32)

                            subject: str = args["image_source_subject"]
                            subject_from_world: Optional[np.ndarray] = vicon.get_segment_global_pose(subject, subject)
                            subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject)
                            if subject_from_world is not None and subject_from_source is not None:
                                world_from_source: np.ndarray = np.linalg.inv(subject_from_world) @ subject_from_source
                                if world_from_initial is None:
                                    world_from_initial = world_from_source
                                initial_from_source: np.ndarray = np.linalg.inv(world_from_initial) @ world_from_source
                                # print(frame_number, initial_from_source)

                                inv_pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                                if inv_pose is not None:
                                    pose = np.linalg.inv(inv_pose)

                                    print(frame_number, initial_from_source, "\n", pose)

                                    # TODO
                                    client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                                        frame_idx, colour_image, ImageUtil.to_short_depth(dummy_depth_image),
                                        pose, msg
                                    ))

                                    # Increment the frame index.
                                    frame_idx += 1

                    # Show the most recent colour image (if any) so that the user can see what's going on.
                    if colour_image is not None:
                        cv2.imshow("Vicon Disk Client", colour_image)

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
