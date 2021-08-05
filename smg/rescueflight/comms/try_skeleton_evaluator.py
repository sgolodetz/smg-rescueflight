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

    # Construct the remote skeleton detector.
    with RemoteSkeletonDetector() as skeleton_detector:
        image: np.ndarray = cv2.imread(args["input_file"])
        world_from_camera: np.ndarray = np.eye(4)

        # Detect the people in the input image.
        skeletons, _ = skeleton_detector.detect_skeletons(image, world_from_camera)

        # TODO
        if skeletons is not None:
            matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]] = [
                [(s, s) for s in skeletons],
                [(s, None) for s in skeletons]
            ]
            evaluator: SkeletonEvaluator = SkeletonEvaluator.make_default()
            correct_keypoint_table: np.ndarray = evaluator.make_correct_keypoint_table(matched_skeletons)
            print(correct_keypoint_table)
            pcks: Dict[str, float] = evaluator.calculate_3d_pcks(correct_keypoint_table)
            print(pcks)


if __name__ == "__main__":
    main()
