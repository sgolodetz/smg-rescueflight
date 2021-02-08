import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional

from smg.pyopencv import CVMat1b
from smg.pyremode import *


def print_se3(se3: SE3f) -> None:
    print()
    for row in range(3):
        print([se3.data(row, col) for col in range(4)])
    print()


def main():
    depthmap: Depthmap = Depthmap(640, 480, 481.2, 319.5, -480.0, 239.5)
    reference_colour_image: Optional[np.ndarray] = None

    _, ax = plt.subplots(2, 2)

    # Read in all of the lines in the trajectory file.
    with open("C:/rpg_open_remode/test_data/first_200_frames_traj_over_table_input_sequence.txt", "r") as f:
        lines = f.read().split("\n")

    for line in lines:
        if not line:
            continue

        filename, tx, ty, tz, qx, qy, qz, qw = line.split(" ")

        colour_image: np.ndarray = cv2.imread(f"C:/rpg_open_remode/test_data/images/{filename}")
        if colour_image is None:
            continue

        cv2.imshow("Image", colour_image)
        cv2.waitKey(1)

        se3: SE3f = SE3f(float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz))
        se3 = se3.inv()
        print_se3(se3)

        grey_image: np.ndarray = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
        cv_grey_image: CVMat1b = CVMat1b.zeros(*grey_image.shape[:2])
        np.copyto(np.array(cv_grey_image, copy=False), grey_image)

        if reference_colour_image is None:
            reference_colour_image = colour_image
            depthmap.set_reference_image(cv_grey_image, se3, 0.1, 4.0)
        else:
            depthmap.update(cv_grey_image, se3)

        estimated_depth_image: np.ndarray = np.array(depthmap.get_denoised_depthmap())

        ax[0, 0].clear()
        ax[0, 1].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()
        ax[0, 0].imshow(reference_colour_image[:, :, [2, 1, 0]])
        ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
        ax[1, 0].imshow(colour_image[:, :, [2, 1, 0]])
        # ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

        plt.draw()
        plt.waitforbuttonpress(0.001)


if __name__ == "__main__":
    main()
