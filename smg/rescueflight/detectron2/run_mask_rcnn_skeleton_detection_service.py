import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from timeit import default_timer as timer
from typing import Callable, List, Optional, Tuple

from smg.comms.skeletons import SkeletonDetectionService
from smg.detectron2 import InstanceSegmenter
from smg.skeletons import Skeleton3D


def make_frame_processor(segmenter: InstanceSegmenter, *, debug: bool = False) -> \
        Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]],
            Tuple[List[Skeleton3D], Optional[np.ndarray]]
        ]:
    """
    Make a frame processor for a skeleton detection service that uses Mask R-CNN to generate people masks.

    :param segmenter:   The Mask R-CNN instance segmenter.
    :param debug:       Whether to print debug messages.
    :return:            The frame processor.
    """
    # noinspection PyUnusedLocal
    def generate_people_mask(frame_idx: int, colour_image: np.ndarray, depth_image: np.ndarray,
                             world_from_camera: np.ndarray, intrinsics: Tuple[float, float, float, float]) \
            -> Tuple[List[Skeleton3D], Optional[np.ndarray]]:
        """
        Generate a people mask for an RGB image using Mask R-CNN.

        :param frame_idx:           Passed in by the skeleton detection service, but ignored.
        :param colour_image:        The RGB image.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   Passed in by the skeleton detection service, but ignored.
        :param intrinsics:          The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:                    An empty list of skeletons and a people mask for the RGB image.
        """
        if debug:
            start = timer()

        # Segment the image.
        instances: List[InstanceSegmenter.Instance] = segmenter.segment(colour_image)

        # Make the people mask by unioning the masks of the instances labelled "person".
        people_mask: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.uint8)
        for i in range(len(instances)):
            instance: InstanceSegmenter.Instance = instances[i]
            if instance.label == "person":
                people_mask |= instance.mask

        # Dilate the people mask to mitigate the "halo effect", in which a halo around each person is fused into
        # the scene representation. A rather large kernel size is needed for this in practice.
        kernel_size: int = 21
        kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
        people_mask = cv2.dilate(people_mask, kernel)

        if debug:
            end = timer()

            # noinspection PyUnboundLocalVariable
            print(f"Segmentation Time: {end - start}s")

        return [], people_mask

    return generate_people_mask


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=7852,
        help="the port on which the service should listen for a connection"
    )
    args: dict = vars(parser.parse_args())

    # Initialise PyGame and create a hidden window so that we can use OpenGL.
    pygame.init()
    window_size: Tuple[int, int] = (1, 1)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    # Construct the Mask R-CNN instance segmenter.
    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()

    # Run the skeleton detection service.
    service: SkeletonDetectionService = SkeletonDetectionService(make_frame_processor(segmenter), args["port"])
    service.run()


if __name__ == "__main__":
    main()
