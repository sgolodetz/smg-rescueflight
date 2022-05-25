import cv2
import glob
import numpy as np
import os

from argparse import ArgumentParser
from timeit import default_timer as timer
from typing import List

from smg.relocalisation.dsacstar_relocaliser import DSACStarRelocaliser
from smg.utility import PoseUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true",
        help="whether to print out debug messages"
    )
    parser.add_argument(
        "--hypothesis_count", type=int, default=64,
        help="the number of RANSAC hypotheses to consider"
    )
    parser.add_argument(
        "--image_height", type=int, default=480,
        help="the height to which the colour images will be rescaled"
    )
    parser.add_argument(
        "--inlier_alpha", type=float, default=100.0,
        help="the alpha parameter to use for soft inlier counting"
    )
    parser.add_argument(
        "--inlier_threshold", type=float, default=10.0,
        help="the inlier threshold to use when sampling RANSAC hypotheses (in pixels)"
    )
    parser.add_argument(
        "--max_pixel_error", type=float, default=100.0,
        help="the maximum reprojection error to use when checking pose consistency (in pixels)"
    )
    parser.add_argument(
        "--network_filename", type=str, required=True,
        help="the name of the file containing the DSAC* network"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="the name of the directory containing the test sequence"
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="whether to load a tiny network to massively reduce the memory footprint"
    )
    args: dict = vars(parser.parse_args())

    # Construct the relocaliser.
    relocaliser: DSACStarRelocaliser = DSACStarRelocaliser(
        args["network_filename"],
        debug=args["debug"],
        hypothesis_count=args["hypothesis_count"],
        image_height=args["image_height"],
        inlier_alpha=args["inlier_alpha"],
        inlier_threshold=args["inlier_threshold"],
        max_pixel_error=args["max_pixel_error"],
        tiny=args["tiny"]
    )

    # For each frame in the test sequence:
    image_dir: str = os.path.join(args["test_dir"], "rgb")
    image_filenames: List[str] = sorted(glob.glob(f"{image_dir}/*.color.png"))
    max_estimation_time_ms: float = 0.0

    for image_filename in image_filenames:
        # Normalise the colour image filename.
        image_filename = image_filename.replace("\\", "/")

        # Determine the ground truth pose filename.
        pose_filename: str = image_filename.replace("/rgb/", "/poses/").replace("color.png", "pose.txt")

        # Load in the image and the ground truth pose.
        image: np.ndarray = cv2.imread(image_filename)
        gt_pose: np.ndarray = PoseUtil.load_pose(pose_filename)

        # Estimate the pose.
        start = timer()
        estimated_pose, scene_coordinates = relocaliser.estimate_pose(image, 525.0)
        end = timer()
        estimation_time_ms: float = (end - start) * 1000
        max_estimation_time_ms = max(estimation_time_ms, max_estimation_time_ms)

        # Print out the estimated and ground truth poses for comparison.
        print(f"=== {image_filename} ===")
        print(f"Current: {estimation_time_ms:.1f}ms; Max: {max_estimation_time_ms:.1f}ms")
        print("--- Estimated ---")
        print(estimated_pose)
        print("--- Ground Truth ---")
        print(gt_pose)

        # Show the scene coordinates image. If the user presses the 'q' key, early out.
        cv2.imshow("Scene Coordinates", cv2.resize(scene_coordinates, (640, 480), interpolation=cv2.INTER_NEAREST))
        c: int = cv2.waitKey(1)
        if c == ord('q'):
            break


if __name__ == "__main__":
    main()
