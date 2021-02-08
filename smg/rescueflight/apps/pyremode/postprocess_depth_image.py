import cv2
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyremode import DepthProcessor
from smg.utility import ImageUtil


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--input_depth_file", "-i", type=str, required=True, help="the input depth image file")
    parser.add_argument("--convergence_file", "-m", type=str, required=True, help="the convergence map file")
    parser.add_argument("--colour_file", "-c", type=str, help="the optional colour image file")
    parser.add_argument("--output_depth_file", "-o", type=str, help="the optional output depth image file")
    args: dict = vars(parser.parse_args())

    # Load the input depth image and convergence map.
    input_depth_image: np.ndarray = ImageUtil.load_depth_image(args["input_depth_file"])
    convergence_map: np.ndarray = cv2.imread(args["convergence_file"], cv2.IMREAD_UNCHANGED)

    # Load the camera intrinsics.
    # FIXME: These should be loaded in rather than hard-coded.
    # intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    intrinsics: Tuple[float, float, float, float] = (946.60441222, 941.38386885, 460.29254907, 357.08431882)

    # Post-process the depth image to reduce noise.
    output_depth_image: np.ndarray = DepthProcessor.postprocess_depth(input_depth_image, convergence_map, intrinsics)

    # If an output file has been specified, save the post-processed depth image to it. Otherwise, visualise it instead.
    output_depth_file: Optional[str] = args.get("output_depth_file")
    if output_depth_file is not None:
        ImageUtil.save_depth_image(output_depth_file, output_depth_image)
    else:
        _, ax = plt.subplots(1, 3)
        ax[0].imshow(input_depth_image, vmin=0.0, vmax=4.0)
        ax[1].imshow(convergence_map)
        ax[2].imshow(output_depth_image, vmin=0.0, vmax=4.0)
        plt.draw()
        plt.waitforbuttonpress()

        colour_file: Optional[str] = args.get("colour_file")
        if colour_file is not None:
            colour_image: np.ndarray = cv2.imread(colour_file)
            VisualisationUtil.visualise_rgbd_image(colour_image, output_depth_image, intrinsics)


if __name__ == "__main__":
    main()
