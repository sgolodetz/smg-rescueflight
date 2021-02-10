import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

from smg.pyremode import DepthProcessor
from smg.utility import ImageUtil


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, required=True, help="the input file")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="the output file")
    args: dict = vars(parser.parse_args())

    # Load in the input depth image.
    input_depth_image: np.ndarray = ImageUtil.load_depth_image(args["input_file"])

    # Densify it using the Delaunay triangulation approach.
    output_depth_image, triangulation = DepthProcessor.densify_depth_delaunay(input_depth_image)

    # Visualise the results.
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(input_depth_image, vmin=0.0, vmax=4.0)
    ax[1].imshow(output_depth_image, vmin=0.0, vmax=4.0)
    plt.gca().invert_yaxis()
    ax[2].triplot(triangulation)
    ax[2].set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
