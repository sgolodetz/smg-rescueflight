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
        "--keep_scale", action="store_true",
        help="whether to keep the existing scale rather than trying to correct it"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="whether to save the estimated scale and world-to-Vicon transformation to disk"
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

    keep_scale: bool = args["keep_scale"]
    save: bool = args["save"]
    sequence_dir: str = args["sequence_dir"]
    source_subject: str = args["source_subject"]

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

    # Read in the trajectories.
    frame_numbers: List[int] = []
    vicon_trajectory: List[np.ndarray] = []
    world_trajectory: List[np.ndarray] = []

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

            # Provided both poses are available, update the trajectories.
            if vicon_from_source is not None and world_from_source is not None:
                frame_numbers.append(frame_number)
                vicon_trajectory.append(vicon_from_source)
                world_trajectory.append(world_from_source)

    # Try to correct the scale if requested to do so.
    scale_factor: float = 1.0
    if not keep_scale:
        vicon_from_previous: Optional[np.ndarray] = None
        world_from_previous: Optional[np.ndarray] = None

        vicon_movement: float = 0.0
        world_movement: float = 0.0

        for i in range(len(vicon_trajectory)):
            vicon_from_source: np.ndarray = vicon_trajectory[i]
            world_from_source: np.ndarray = world_trajectory[i]

            if world_from_previous is not None and vicon_from_previous is not None:
                vicon_movement += np.linalg.norm(vicon_from_source[0:3, 3] - vicon_from_previous[0:3, 3])
                world_movement += np.linalg.norm(world_from_source[0:3, 3] - world_from_previous[0:3, 3])

            vicon_from_previous = vicon_from_source.copy()
            world_from_previous = world_from_source.copy()

        if world_movement > 0.0:
            scale_factor = vicon_movement / world_movement

    # Initialise the list of world-to-Vicon transformation estimates.
    vicon_from_world_estimates: List[np.ndarray] = []

    # For each pair of corresponding poses in the trajectories:
    for i in range(len(vicon_trajectory)):
        vicon_from_source: np.ndarray = vicon_trajectory[i]
        world_from_source: np.ndarray = world_trajectory[i]

        # Apply the estimated scale factor to the pose calculated by the tracker.
        world_from_source[0:3, 3] *= scale_factor

        # Add an estimate of the world-to-Vicon transformation to the list.
        vicon_from_world_estimates.append(vicon_from_source @ np.linalg.inv(world_from_source))

        # Print out a message to show progress.
        print(f"Added estimate for frame {frame_numbers[i]}...")

    # Print out the estimated scale factor.
    print()
    print(f"Scale factor: {scale_factor}")

    # Calculate and print the final estimate of the world-to-Vicon transformation.
    vicon_from_world: np.ndarray = GeometryUtil.blend_rigid_transforms(vicon_from_world_estimates)
    print()
    print("vTw:")
    print(vicon_from_world)

    # Save both to disk if requested.
    if save:
        reconstruction_dir: str = os.path.join(sequence_dir, "reconstruction")
        os.makedirs(reconstruction_dir, exist_ok=True)

        scale_filename: str = os.path.join(reconstruction_dir, "vicon_from_world_scale_factor.txt")
        with open(scale_filename, "w") as file:
            file.write(f"{scale_factor}\n")

        vicon_from_world_filename: str = os.path.join(reconstruction_dir, "vicon_from_world.txt")
        PoseUtil.save_pose(vicon_from_world_filename, vicon_from_world)


if __name__ == "__main__":
    main()
