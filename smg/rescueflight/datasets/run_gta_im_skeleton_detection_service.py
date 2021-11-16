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


def generate_id_map(map_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Make an instance ID image of the specified size by loading one from disk and resizing it.

    .. note::
        This function is adapted from gta_utils.py in the GTA-IM code.

    :param map_path:    The path to the instance ID image on disk.
    :param image_size:  The desired size of the output image.
    :return:            The resized instance ID image.
    """
    id_map = cv2.imread(map_path, -1)
    id_map = cv2.resize(id_map, image_size, interpolation=cv2.INTER_NEAREST)
    h, w, _ = id_map.shape
    id_map = np.concatenate(
        (id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2
    )
    id_map.dtype = np.uint32
    return np.squeeze(id_map)


def make_keypoint(new_name: str, old_name: str, frame_info: Dict[str, Any], origin: np.ndarray) -> Tuple[str, Keypoint]:
    """
    Make a skeleton keypoint from its known position in the GTA-IM dataset.

    .. note::
        In practice, the origin will just be the initial position of the camera.

    :param new_name:    The name we want to give the keypoint in our system.
    :param old_name:    The name of the keypoint in the GTA-IM dataset.
    :param frame_info:  The info for the frame (obtained from the GTA-IM .pickle file for the sequence).
    :param origin:      The origin of the coordinate system in which we want to express the keypoint.
    :return:            The keypoint.
    """
    # Transform the keypoint into the desired coordinate system. Note that the rotation here mirrors the rotation
    # of the camera pose in try_load_frame in the GTA-IM client.
    position: np.ndarray = np.array(frame_info[old_name]) - origin
    position = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]) @ position

    return new_name, Keypoint(new_name, position)


def make_sequence_key(sequence_dir: str) -> str:
    """
    Make the key for a GTA-IM sequence based on the path to the sequence directory.

    .. note::
        As an example, the sequence key for .../gta-im/FPS-5/2020-06-09-17-07-15 is FPS-5/2020-06-09-17-07-15.

    :param sequence_dir:    The directory containing the GTA-IM sequence.
    :return:                The key for the sequence.
    """
    components: List[str] = os.path.normpath(sequence_dir).split(os.sep)
    return f"{components[-2]}/{components[-1]}"


def make_frame_processor(sequence_dir: str, info: List[Dict[str, Any]], info_npz: np.lib.npyio.NpzFile) -> \
        Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]],
            Tuple[List[Skeleton3D], Optional[np.ndarray]]
        ]:
    """
    Make a frame processor for a skeleton detection service that yields ground-truths 3D skeletons and people masks
    loaded from a GTA-IM sequence on disk.

    :param sequence_dir:    The directory containing the GTA-IM sequence.
    :param info:            The information loaded from the sequence's .pickle file.
    :param info_npz:        The information loaded from the sequence's .npz file.
    :return:                The frame processor.
    """
    # noinspection PyUnusedLocal
    def get_skeletons(frame_idx: int, colour_image: np.ndarray, depth_image: np.ndarray, world_from_camera: np.ndarray,
                      intrinsics: Tuple[float, float, float, float]) \
            -> Tuple[List[Skeleton3D], Optional[np.ndarray]]:
        """
        Load and yield the ground-truth 3D skeletons and people mask for the specified GTA-IM frame.

        :param frame_idx:           The frame index.
        :param colour_image:        Passed in by the skeleton detection service, but ignored.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   Passed in by the skeleton detection service, but ignored.
        :param intrinsics:          Passed in by the skeleton detection service, but ignored.
        :return:                    The ground-truth 3D skeletons and people mask for the specified GTA-IM frame.
        """
        # Get the frame info.
        frame_info: Dict[str, Any] = info[frame_idx]

        # Construct the skeleton.
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

        # Construct the people mask.
        height, width = colour_image.shape[:2]
        id_map: np.ndarray = generate_id_map(os.path.join(sequence_dir, f"{frame_idx:05d}_id.png"), (width, height))

        # Note: The instance IDs corresponding to people don't seem to be available in the sequence information, so
        #       I'm hard-coding them here for specific sequences. This will change when new sequences are added.
        sequence_to_person_ids: Dict[str, List[int]] = {
            "FPS-5/2020-06-09-16-09-56": [260354, 512770],
            "FPS-5/2020-06-09-17-14-03": [16386, 389890],
            "FPS-5/2020-06-10-21-53-42": [16386],
            "FPS-5/2020-06-12-18-53-02": [16386],
            "FPS-5/2020-06-21-19-42-55": [609026],
            "FPS-30/2020-06-02-18-09-25": [465415, 725762],
            "FPS-30/2020-06-03-20-28-01": [223239]
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

        return [skeleton], people_mask

    return get_skeletons


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--frame_idx", type=int,
        help="an optional frame index for which to show the instance masks (instead of running the service)"
    )
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

    # Check whether an optional frame index has been specified.
    frame_idx: Optional[int] = args.get("frame_idx")

    # If it has:
    if frame_idx is not None:
        # Generate the instance ID image for the frame.
        id_map: np.ndarray = generate_id_map(os.path.join(sequence_dir, f"{frame_idx:05d}_id.png"), (1920, 1080))

        # Show a binary mask for each unique instance ID, so that we can figure out which is which.
        ids: np.ndarray = np.unique(id_map)
        for i in ids:
            cv2.imshow(f"ID {i}", cv2.resize(np.where(id_map == i, 255, 0).astype(np.uint8), (0, 0), fx=0.25, fy=0.25))

        # Wait for the user to press a key, and then exit.
        cv2.waitKey()
        exit(0)

    # Run the skeleton detection service.
    service: SkeletonDetectionService = SkeletonDetectionService(
        make_frame_processor(sequence_dir, info, info_npz), args["port"]
    )
    service.run()


if __name__ == "__main__":
    main()
