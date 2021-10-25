import cv2
import numpy as np
import os
import pickle

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from smg.comms.skeletons import SkeletonDetectionService
from smg.skeletons import Keypoint, Skeleton3D


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


def make_keypoint(new_name: str, old_name: str, frame_info: Dict[str, Any], origin: np.ndarray) -> Tuple[str, Keypoint]:
    position: np.ndarray = np.array(frame_info[old_name]) - origin
    position = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]) @ position
    return new_name, Keypoint(new_name, position)


def make_sequence_key(sequence_dir: str) -> str:
    sequence_components: List[str] = os.path.normpath(sequence_dir).split(os.sep)
    return f"{sequence_components[-2]}/{sequence_components[-1]}"


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
        frame_info: Dict[str, Any] = info[frame_idx]

        world_from_initial: np.ndarray = np.linalg.inv(np.transpose(info_npz["world2cam_trans"][0]))
        origin: np.ndarray = world_from_initial[0:3, 3]

        keypoints: Dict[str, Keypoint] = dict([
            make_keypoint("Head", "head", frame_info, origin),
            make_keypoint("LAnkle", "left_ankle", frame_info, origin),
            make_keypoint("LClavicle", "left_clavicle", frame_info, origin),
            make_keypoint("LElbow", "left_elbow", frame_info, origin),
            make_keypoint("LHip", "left_hip", frame_info, origin),
            make_keypoint("LKnee", "left_knee", frame_info, origin),
            make_keypoint("LShoulder", "left_shoulder", frame_info, origin),
            make_keypoint("LWrist", "left_wrist", frame_info, origin),
            make_keypoint("MidHip", "spine4", frame_info, origin),
            make_keypoint("Neck", "neck", frame_info, origin),
            make_keypoint("RAnkle", "right_ankle", frame_info, origin),
            make_keypoint("RClavicle", "right_clavicle", frame_info, origin),
            make_keypoint("RElbow", "right_elbow", frame_info, origin),
            make_keypoint("RHip", "right_hip", frame_info, origin),
            make_keypoint("RKnee", "right_knee", frame_info, origin),
            make_keypoint("RShoulder", "right_shoulder", frame_info, origin),
            make_keypoint("RWrist", "right_wrist", frame_info, origin),
            make_keypoint("Spine0", "spine0", frame_info, origin),
            make_keypoint("Spine1", "spine1", frame_info, origin),
            make_keypoint("Spine2", "spine2", frame_info, origin),
            make_keypoint("Spine3", "spine3", frame_info, origin)
        ])

        keypoint_names: Dict[int, str] = {
            0: "Head", 1: "Neck", 2: "RClavicle", 3: "RShoulder", 4: "RElbow", 5: "RWrist",
            6: "LClavicle", 7: "LShoulder", 8: "LElbow", 9: "LWrist", 10: "Spine0",
            11: "Spine1", 12: "Spine2", 13: "Spine3", 14: "MidHip", 15: "RHip",
            16: "RKnee", 17: "RAnkle", 18: "LHip", 19: "LKnee", 20: "LAnkle"
        }

        keypoint_pairs: List[Tuple[str, str]] = [
            (keypoint_names[i], keypoint_names[j]) for i, j in [
                # See: https://github.com/ZheC/GTA-IM-Dataset.
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8), (8, 9), (1, 10), (10, 11),
                (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (14, 18), (18, 19), (19, 20)
            ]
        ]

        skeleton: Skeleton3D = Skeleton3D(keypoints, keypoint_pairs)

        height, width = colour_image.shape[:2]
        id_map: np.ndarray = generate_id_map(os.path.join(sequence_dir, f"{frame_idx:05d}_id.png"), (width, height))

        sequence_to_person_ids: Dict[str, List[int]] = {
            "FPS-5/2020-06-09-17-14-03": [16386, 389890],
            "FPS-5/2020-06-10-21-53-42": [16386],
            "FPS-5/2020-06-21-19-42-55": [609026],
            "FPS-30/2020-06-02-18-09-25": [465415, 725762]
        }

        person_ids: List[int] = sequence_to_person_ids.get(make_sequence_key(sequence_dir), [])

        people_mask: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.uint8)
        for person_id in person_ids:
            person_mask: np.ndarray = np.where(id_map == person_id, 255, 0).astype(np.uint8)
            people_mask |= person_mask

        # Dilate the people mask to mitigate the "halo effect", in which a halo around each person is fused into
        # the scene representation. Since the masks are ground-truth ones, only a small kernel size is needed.
        kernel_size: int = 5
        kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
        people_mask = cv2.dilate(people_mask, kernel)

        # cv2.imshow("People Mask", people_mask)
        # cv2.waitKey(1)

        return [skeleton], people_mask

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
    # idm = generate_id_map(os.path.join(sequence_dir, "00000_id.png"), (1920, 1080))
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
