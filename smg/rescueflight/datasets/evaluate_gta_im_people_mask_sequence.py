import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import List, Optional


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray, *, debug: bool = False) -> Optional[float]:
    mask_i: np.ndarray = np.logical_and(mask1, mask2).astype(np.uint8) * 255
    mask_u: np.ndarray = np.logical_or(mask1, mask2).astype(np.uint8) * 255

    if debug:
        cv2.imshow("Intersection", mask_i)
        cv2.imshow("Union", mask_u)

    i: int = np.count_nonzero(mask_i)
    u: int = np.count_nonzero(mask_u)

    return i / u if u > 0 else None


def get_frame_index(filename: str) -> int:
    """
    Get the frame index corresponding to a file containing a people mask image.

    .. note::
        The files are named <frame index>.people.png, so we can get the frame indices directly from the file names.

    :param filename:    The name of a file containing a people mask image.
    :return:            The corresponding frame index.
    """
    frame_idx, _, _ = filename.split(".")
    return int(frame_idx)


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true",
        help="whether to enable debugging"
    )
    parser.add_argument(
        "--generator_type", "-t", type=str, default="maskrcnn", choices=("gt", "lcrnet", "maskrcnn"),
        help="the people mask generator whose (pre-saved) masks are to be evaluated"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the sequence"
    )
    args: dict = vars(parser.parse_args())

    debug: bool = args["debug"]
    generator_type: str = args["generator_type"]
    sequence_dir: str = args["sequence_dir"]

    # Determine the list of people mask filenames to use.
    mask_filenames: List[str] = [
        f for f in os.listdir(os.path.join(sequence_dir, "people", "gt")) if f.endswith(".people.png")
    ]

    mask_filenames = sorted(mask_filenames, key=get_frame_index)

    # Initialise some variables.
    count: int = 0
    f1_sum: float = 0.0
    iou_sum: float = 0.0

    # For each frame in the sequence:
    for i in range(len(mask_filenames)):
        # Load in the ground-truth mask.
        gt_mask: Optional[np.ndarray] = cv2.imread(
            os.path.join(sequence_dir, "people", "gt", mask_filenames[i])
        )

        # Load in the generated mask.
        generated_mask: Optional[np.ndarray] = cv2.imread(
            os.path.join(sequence_dir, "people", generator_type, mask_filenames[i])
        )

        # Assuming the ground-truth mask is available (which should always be the case, unless it gets deleted):
        if gt_mask is not None:
            # TODO
            gt_mask = gt_mask[:, :, 0]

            if debug:
                cv2.imshow("GT Mask", gt_mask)

            if generated_mask is not None:
                generated_mask = generated_mask[:, :, 0]

                # BEGIN TEMPORARY
                # This is for variety, since I can't run Mask R-CNN on the desktop in the lab yet.
                generated_mask = np.flipud(generated_mask)
                # END TEMPORARY

                if debug:
                    cv2.imshow("Generated Mask", generated_mask)

                iou: Optional[float] = calculate_iou(generated_mask, gt_mask, debug=debug)

                if iou is not None:
                    f1: float = 2 * iou / (iou + 1)

                    if debug:
                        print(f"Frame {get_frame_index(mask_filenames[i])} - IoU: {iou}; F1: {f1}")

                    count += 1
                    f1_sum += f1
                    iou_sum += iou
            else:
                pass

            if debug:
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

    if debug:
        print()

    print(f"Frame Count: {count}")
    print(f"Mean IoU: {iou_sum / count}")
    print(f"Mean F1: {f1_sum / count}")


if __name__ == "__main__":
    main()
