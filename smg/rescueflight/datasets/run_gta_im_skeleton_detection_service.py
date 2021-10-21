import cv2
import numpy as np
import os
import pickle

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from smg.comms.skeletons import SkeletonDetectionService
from smg.skeletons import Skeleton3D


# def generate_id_map(map_path):
#     # FIXME: This function is currently borrowed from gta_utils.py in the GTA-IM code, but should really be imported.
#     id_map = cv2.imread(map_path, -1)
#     h, w, _ = id_map.shape
#     id_map = np.concatenate(
#         (id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2
#     )
#     id_map.dtype = np.uint32
#     return id_map

def generate_id_map(map_path, image_size):
    # FIXME: This function is adapted from gta_utils.py in the GTA-IM code.
    id_map = cv2.imread(map_path, -1)
    id_map = cv2.resize(id_map, image_size, interpolation=cv2.INTER_NEAREST)
    h, w, _ = id_map.shape
    id_map = np.concatenate(
        (id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2
    )
    id_map.dtype = np.uint32
    return np.squeeze(id_map)


def make_frame_processor(sequence_dir: str, info: List[Dict[str, Any]], info_npz: np.lib.npyio.NpzFile) -> \
        Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]],
            Tuple[List[Skeleton3D], Optional[np.ndarray]]
        ]:
    """
    Make a frame processor for a skeleton detection service that yields ground-truths 3D skeletons and people masks
    loaded from a GTA-IM sequence on disk.

    :param sequence_dir:    TODO
    :param info:            TODO
    :param info_npz:        TODO
    :return:                The frame processor.
    """
    # noinspection PyUnusedLocal
    def get_skeletons(frame_idx: int, colour_image: np.ndarray, depth_image: np.ndarray, world_from_camera: np.ndarray,
                      intrinsics: Tuple[float, float, float, float]) \
            -> Tuple[List[Skeleton3D], Optional[np.ndarray]]:
        """
        Load and yield the ground-truth 3D skeletons and people mask for the specified GTA-IM frame.

        :param frame_idx:           TODO
        :param colour_image:        Passed in by the skeleton detection service, but ignored.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   Passed in by the skeleton detection service, but ignored.
        :param intrinsics:          Passed in by the skeleton detection service, but ignored.
        :return:                    The ground-truth 3D skeletons and people mask for the specified GTA-IM frame.
        """
        height, width = colour_image.shape[:2]
        id_map: np.ndarray = generate_id_map(os.path.join(sequence_dir, f"{frame_idx:05d}_id.png"), (width, height))

        person_ids: List[int] = [16386, 389890]

        people_mask: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.uint8)
        for person_id in person_ids:
            person_mask: np.ndarray = np.where(id_map == person_id, 255, 0).astype(np.uint8)
            people_mask |= person_mask

        cv2.imshow("People Mask", people_mask)
        cv2.waitKey(1)

        return [], people_mask

    return get_skeletons


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=7852,
        help="the port on which the service should listen for a connection"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    # Initialise PyGame and create a hidden window so that we can use OpenGL.
    pygame.init()
    window_size: Tuple[int, int] = (1, 1)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    # Load in the sequence information, as per https://github.com/ZheC/GTA-IM-Dataset.
    sequence_dir: str = args["sequence_dir"]
    info: List[Dict[str, Any]] = pickle.load(open(os.path.join(sequence_dir, "info_frames.pickle"), "rb"))
    info_npz: np.lib.npyio.NpzFile = np.load(os.path.join(sequence_dir, "info_frames.npz"))

    # TEMPORARY
    # idm = generate_id_map(os.path.join(sequence_dir, "00000_id.png"))
    # ids = np.unique(idm)
    # print(ids)
    # for id in ids:
    #     cv2.imshow(f"ID {id}", cv2.resize(np.where(idm == id, 255, 0).astype(np.uint8), (0, 0), fx=0.25, fy=0.25))
    # cv2.waitKey()
    # exit(0)

    # Run the skeleton detection service.
    service: SkeletonDetectionService = SkeletonDetectionService(
        make_frame_processor(sequence_dir, info, info_npz), args["port"]
    )
    service.run()


if __name__ == "__main__":
    main()
