import cv2
import numpy as np

from argparse import ArgumentParser

from smg.relocalisation.dsacstar_relocaliser import DSACStarRelocaliser
from smg.utility import PoseUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--hypothesis_count", type=int, default=64,
        help="TODO"
    )
    parser.add_argument(
        "--inlier_alpha", type=float, default=100.0,
        help="TODO"
    )
    parser.add_argument(
        "--inlier_threshold", type=float, default=10.0,
        help="TODO"
    )
    parser.add_argument(
        "--max_pixel_error", type=float, default=100.0,
        help="TODO"
    )
    parser.add_argument(
        "--network_filename", type=str, required=True,
        help="TODO"
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="TODO"
    )
    args: dict = vars(parser.parse_args())

    relocaliser: DSACStarRelocaliser = DSACStarRelocaliser(
        hypothesis_count=args["hypothesis_count"],
        inlier_alpha=args["inlier_alpha"],
        inlier_threshold=args["inlier_threshold"],
        max_pixel_error=args["max_pixel_error"],
        network_filename=args["network_filename"],
        tiny=args["tiny"]
    )

    image: np.ndarray = cv2.imread("C:/smglib/smg-relocalisation/smg/external/dsacstar/datasets/7scenes_heads/test/rgb/seq-01-frame-000000.color.png")
    print(relocaliser.estimate_pose(image, 525.0))
    print(PoseUtil.load_pose("C:/smglib/smg-relocalisation/smg/external/dsacstar/datasets/7scenes_heads/test/poses/seq-01-frame-000000.pose.txt"))


if __name__ == "__main__":
    main()
