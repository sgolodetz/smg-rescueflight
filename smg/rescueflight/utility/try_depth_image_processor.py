import cv2
import numpy as np

from timeit import default_timer as timer

from smg.utility import DepthImageProcessor, ImageUtil


def main() -> None:
    depth_image: np.ndarray = ImageUtil.load_depth_image("D:/jiaxing/frame-000000.depth.png")

    start = timer()

    segmentation, stats, depth_edges = DepthImageProcessor.segment_depth_image(
        depth_image, threshold=0.02
    )

    end = timer()
    print(end - start)

    start = timer()

    depth_image, segmentation = DepthImageProcessor.remove_isolated_regions(
        depth_image, segmentation, stats, min_region_size=500
    )

    end = timer()
    print(end - start)

    cv2.imshow("Depth Image", depth_image / 2)
    cv2.imshow("Depth Edges", depth_edges)
    cv2.imshow("Segmentation", ImageUtil.colourise_segmentation(segmentation))
    cv2.waitKey()


if __name__ == "__main__":
    main()
