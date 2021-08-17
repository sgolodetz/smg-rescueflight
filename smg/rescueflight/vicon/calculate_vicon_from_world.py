import numpy as np
import os

from argparse import ArgumentParser
from typing import List, Optional

from smg.utility import GeometryUtil, PoseUtil
from smg.vicon import OfflineViconInterface, SubjectFromSourceCache


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--save", action="store_true",
        help="whether to save the estimated world-to-Vicon transformation to disk"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    parser.add_argument(
        "--source_subject", type=str, default="Tello",
        help="the name of the Vicon subject corresponding to the image source"
    )
    args: dict = vars(parser.parse_args())

    save: bool = args["save"]
    sequence_dir: str = args["sequence_dir"]
    source_subject: str = args["source_subject"]

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

    # Initialise the list of world-to-Vicon transformation estimates.
    vicon_from_world_estimates: List[np.ndarray] = []

    # Connect to the Vicon interface.
    with OfflineViconInterface(folder=sequence_dir) as vicon:
        # Until we run out of Vicon frames to load:
        while vicon.get_frame():
            # Get the frame number associated with the current frame.
            frame_number: int = vicon.get_frame_number()

            # Determine the image source pose according to the Vicon system.
            vicon_from_source: Optional[np.ndarray] = vicon.get_image_source_pose(
                source_subject, subject_from_source_cache
            )

            # Load in the corresponding image source pose calculated by the tracker.
            pose_filename: str = os.path.join(sequence_dir, f"{frame_number}.pose.txt")
            world_from_source: Optional[np.ndarray] = PoseUtil.load_pose(pose_filename) \
                if os.path.exists(pose_filename) else None

            # Provided both poses are available:
            if vicon_from_source is not None and world_from_source is not None:
                # Add an estimate of the world-to-Vicon transformation to the list.
                vicon_from_world_estimates.append(vicon_from_source @ np.linalg.inv(world_from_source))

                # Print out a message to show progress.
                print(f"Added estimate for frame {frame_number}...")

        # At the end of the process, calculate and print the final estimate of the world-to-Vicon transformation.
        vicon_from_world: np.ndarray = GeometryUtil.blend_rigid_transforms(vicon_from_world_estimates)
        print("vTw:")
        print(vicon_from_world)

        # Also save it to disk if requested.
        if save:
            reconstruction_dir: str = os.path.join(sequence_dir, "reconstruction")
            os.makedirs(reconstruction_dir, exist_ok=True)
            vicon_from_world_filename: str = os.path.join(reconstruction_dir, "vicon_from_world.txt")
            PoseUtil.save_pose(vicon_from_world_filename, vicon_from_world)


if __name__ == "__main__":
    main()
