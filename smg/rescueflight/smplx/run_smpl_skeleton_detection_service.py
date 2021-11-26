import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from timeit import default_timer as timer
from typing import Callable, List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector, SkeletonDetectionService
from smg.skeletons import PeopleMaskRenderer, Skeleton3D
from smg.smplx import SMPLBody


def make_frame_processor(skeleton_detector: RemoteSkeletonDetector, people_mask_renderer: PeopleMaskRenderer,
                         smpl_body: SMPLBody, *, debug: bool = False) -> \
        Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]],
            Tuple[List[Skeleton3D], Optional[np.ndarray]]
        ]:
    """
    Make a frame processor for a skeleton detection service that uses an underlying skeleton detector and SMPL
    to generate people masks.

    :param skeleton_detector:       The underlying skeleton detector.
    :param people_mask_renderer:    The people mask renderer.
    :param smpl_body:               The SMPL body model.
    :param debug:                   Whether to print debug messages.
    :return:                        The frame processor.
    """
    # noinspection PyUnusedLocal
    def generate_people_mask(frame_idx: int, colour_image: np.ndarray, depth_image: np.ndarray,
                             world_from_camera: np.ndarray, intrinsics: Tuple[float, float, float, float]) \
            -> Tuple[List[Skeleton3D], Optional[np.ndarray]]:
        """
        Generate a people mask for an RGB image using an underlying skeleton detector and SMPL.

        :param frame_idx:           Passed in by the skeleton detection service, but ignored.
        :param colour_image:        The RGB image.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   Passed in by the skeleton detection service, but ignored.
        :param intrinsics:          The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:                    An empty list of skeletons and a people mask for the RGB image.
        """
        if debug:
            start = timer()

        # Detect the skeletons in the image using the underlying skeleton detector.
        height, width = colour_image.shape[:2]
        skeleton_detector.set_calibration((width, height), intrinsics)
        skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera, frame_idx=frame_idx)

        # Render the people mask.
        people_mask: np.ndarray = people_mask_renderer.render_people_mask(
            render_person_mask, skeletons, world_from_camera, intrinsics, width, height
        )

        if debug:
            end = timer()

            # noinspection PyUnboundLocalVariable
            print(f"People Mask Generation Time: {end - start}s")

        return skeletons, people_mask

    def render_person_mask(skeleton: Skeleton3D) -> None:
        """
        TODO

        :param skeleton:    TODO
        :return:            TODO
        """
        smpl_body.render_from_skeleton(skeleton, colour=(1.0, 1.0, 1.0))

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

    # Load in the SMPL body model.
    smpl_body: SMPLBody = SMPLBody("male")

    # Construct the people mask renderer.
    people_mask_renderer: PeopleMaskRenderer = PeopleMaskRenderer()

    # Construct the remote skeleton detector.
    with RemoteSkeletonDetector(("127.0.0.1", 7854)) as skeleton_detector:
        # Run the skeleton detection service.
        service: SkeletonDetectionService = SkeletonDetectionService(
            make_frame_processor(skeleton_detector, people_mask_renderer, smpl_body), args["port"]
        )
        service.run()


if __name__ == "__main__":
    main()
