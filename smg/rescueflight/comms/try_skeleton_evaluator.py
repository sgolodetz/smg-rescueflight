import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.skeletons import Skeleton3D, SkeletonEvaluator


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=str, default="D:/LCRNet_v2.0/skeleton.png", help="the input image"
    )
    args: dict = vars(parser.parse_args())

    # Make the skeleton evaluator.
    evaluator: SkeletonEvaluator = SkeletonEvaluator.make_default()

    # Load the input image.
    image: np.ndarray = cv2.imread(args["input_file"])

    # Specify a dummy pose.
    world_from_camera: np.ndarray = np.eye(4)

    # Construct the remote skeleton detector.
    with RemoteSkeletonDetector() as skeleton_detector:
        # Detect the people in the input image.
        skeletons, _ = skeleton_detector.detect_skeletons(image, world_from_camera)

        # If any skeletons were detected:
        if skeletons is not None:
            # Make a dummy list of matched skeletons so we can try out the skeleton evaluator.
            matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]] = [
                [(s, s) for s in skeletons],
                [(s, None) for s in skeletons]
            ]

            # Make the per-joint position error table, and print it out.
            per_joint_error_table: np.ndarray = evaluator.make_per_joint_error_table(matched_skeletons)
            print(per_joint_error_table)

            # Calculate the mean per-joint position errors, and print them out.
            mpjpes: Dict[str, float] = evaluator.calculate_mpjpes(per_joint_error_table)
            print(mpjpes)

            # Make the correct keypoint table, and print it out.
            correct_keypoint_table: np.ndarray = SkeletonEvaluator.make_correct_keypoint_table(per_joint_error_table)
            print(correct_keypoint_table)

            # Calculate the percentages of correct keypoints, and print them out.
            pcks: Dict[str, float] = evaluator.calculate_pcks(correct_keypoint_table)
            print(pcks)


if __name__ == "__main__":
    main()
