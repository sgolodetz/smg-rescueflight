import cv2
import numpy as np

from argparse import ArgumentParser
from timeit import default_timer as timer
from typing import List, Optional

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.skeletons import Skeleton


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

        # Repeatedly detect the people in the input image and print out some debug messages.
        while True:
            start = timer()
            skeletons: Optional[List[Skeleton]] = skeleton_detector.detect_skeletons(image, world_from_camera)
            end = timer()

            if skeletons is not None:
                print(f"Remote Detection Time: {end - start}s")
                print(f"Skeletons: {skeletons}")
                print("===")


if __name__ == "__main__":
    main()
