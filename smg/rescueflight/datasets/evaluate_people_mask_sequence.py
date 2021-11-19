import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import List, Optional

from smg.utility import ImageUtil


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
        "--generator_tag", "-t", type=str, required=True,
        help="the tag of the people mask generator whose (pre-saved) masks are to be evaluated"
    )
    parser.add_argument(
        "--gt_generator_tag", type=str, default="gt",
        help="the tag of the (pre-saved) ground-truth masks"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the sequence"
    )
    args: dict = vars(parser.parse_args())

    debug: bool = args["debug"]
    generator_tag: str = args["generator_tag"]
    gt_generator_tag: str = args["gt_generator_tag"]
    sequence_dir: str = args["sequence_dir"]

    # Determine the list of people mask filenames to use.
    mask_filenames: List[str] = [
        f for f in os.listdir(os.path.join(sequence_dir, "people", gt_generator_tag)) if f.endswith(".people.png")
    ]

    mask_filenames = sorted(mask_filenames, key=get_frame_index)

    # Initialise some variables.
    f1_sum: float = 0.0
    iog_count: int = 0
    iog_sum: float = 0.0
    iou_count: int = 0
    iou_sum: float = 0.0

    # For each frame in the sequence:
    for i in range(len(mask_filenames)):
        # Try to load in the ground-truth mask.
        gt_mask: Optional[np.ndarray] = cv2.imread(
            os.path.join(sequence_dir, "people", gt_generator_tag, mask_filenames[i])
        )

        # Try to load in the generated mask.
        generated_mask: Optional[np.ndarray] = cv2.imread(
            os.path.join(sequence_dir, "people", generator_tag, mask_filenames[i])
        )

        # Assuming the masks are available (which should always be the case, unless they've been deleted):
        if gt_mask is not None and generated_mask is not None:
            # Convert them into single-channel images.
            gt_mask = gt_mask[:, :, 0]
            generated_mask = generated_mask[:, :, 0]

            # If the "generated" mask is the ground-truth one, flip it. This is useful for performing quick
            # non-trivial tests on machines where the real people masks are unavailable.
            if generator_tag == gt_generator_tag:
                generated_mask = np.flipud(generated_mask)

            # If we're debugging, show the masks.
            if debug:
                cv2.imshow("GT Mask", gt_mask)
                cv2.imshow("Generated Mask", generated_mask)

            # Try to calculate the IoU between the masks.
            iou: Optional[float] = ImageUtil.calculate_iou(generated_mask, gt_mask, debug=debug)

            # If this succeeds:
            if iou is not None:
                # Compute the F1 score from the IoU.
                f1: float = 2 * iou / (iou + 1)

                # If we're debugging, print out the IoU and F1 scores.
                if debug:
                    print(f"Frame {get_frame_index(mask_filenames[i])} - IoU: {iou}; F1: {f1}")

                # Update the running IoU and F1 sums.
                iou_count += 1
                f1_sum += f1
                iou_sum += iou

            # Try to calculate the IoG (a.k.a. ground-truth coverage ratio) of the "generated" mask.
            iog: Optional[float] = ImageUtil.calculate_iog(generated_mask, gt_mask)

            # If this succeeds:
            if iog is not None:
                # Update the running IoG sum.
                iog_count += 1
                iog_sum += iog

            # If we're debugging, run the OpenCV event loop, and quit if 'q' is pressed.
            if debug:
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

    # If we're debugging, print a blank line before the summary metrics.
    if debug:
        print()

    # Print out the summary metrics. Note that the IoG and IoU counts will be 0 iff the ground-truth masks are blank.
    # If this happens, set the person IDs for the sequence in the GTA-IM skeleton detection service.
    print(f"IoG Frame Count: {iog_count}")
    print(f"IoU Frame Count: {iou_count}")

    if iog_count > 0:
        print(f"Mean IoG: {iog_sum / iog_count}")
    else:
        raise RuntimeError("Error: IoG count is zero (try setting the person IDs in the skeleton detection service)")

    if iou_count > 0:
        print(f"Mean IoU: {iou_sum / iou_count}")
        print(f"Mean F1: {f1_sum / iou_count}")
    else:
        raise RuntimeError("Error: IoU count is zero (try setting the person IDs in the skeleton detection service)")


if __name__ == "__main__":
    main()
