import cv2
import numpy as np
import os
import pickle

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.utility import CameraParameters, GeometryUtil, ImageUtil, PooledQueue


def read_depthmap(name, cam_near_clip, cam_far_clip):
    # FIXME: This function is currently borrowed from gta_utils.py in the GTA-IM code, but should really be imported.
    depth = cv2.imread(name)
    depth = np.concatenate(
        (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
    )
    depth.dtype = np.uint32
    depth = 0.05 * 1000 / depth.astype('float')
    depth = (
        cam_near_clip
        * cam_far_clip
        / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
    )
    return depth


def try_load_frame(frame_idx: int, sequence_dir: str, info: List[Dict[str, Any]], info_npz: np.lib.npyio.NpzFile) \
        -> Optional[Dict[str, Any]]:
    """
    Try to load a frame from a GTA-IM sequence.

    .. note::
        The RGB-D frame is returned as a mapping from strings to pieces of data:
            "colour_image" -> np.ndarray (an 8UC3 image)
            "depth_image" -> np.ndarray (a float image)
            "world_from_camera" -> np.ndarray (a 4x4 transformation matrix)

    :param frame_idx:       The frame index.
    :param sequence_dir:    The sequence directory.
    :param info:            TODO
    :param info_npz:        TODO
    :return:                The RGB-D frame, if possible, or None otherwise.
    """
    # Determine the names of the colour image,  depth image and pose files.
    colour_filename: str = os.path.join(sequence_dir, f"{frame_idx:05d}.jpg")
    depth_filename: str = os.path.join(sequence_dir, f"{frame_idx:05d}.png")

    # If one of the files doesn't exist, early out.
    if not os.path.exists(colour_filename) or not os.path.exists(depth_filename):
        return None

    # Otherwise, load and return the frame.
    frame_info: Dict[str, Any] = info[frame_idx]

    colour_image: np.ndarray = cv2.imread(colour_filename)

    cam_near_clip: float = frame_info["cam_near_clip"]
    cam_far_clip: float = frame_info.get("cam_far_clip", 800.0)
    depth_image: np.ndarray = np.squeeze(read_depthmap(depth_filename, cam_near_clip, cam_far_clip))

    world_from_camera: np.ndarray = np.linalg.inv(np.transpose(info_npz["world2cam_trans"][frame_idx]))
    initial_from_world: np.ndarray = np.transpose(info_npz["world2cam_trans"][0])
    # initial_from_camera: np.ndarray = initial_from_world @ world_from_camera
    initial_from_camera: np.ndarray = world_from_camera.copy()
    initial_from_camera[0:3, 3] -= np.linalg.inv(initial_from_world)[0:3, 3]

    initial_from_camera = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]) @ initial_from_camera

    return {
        "colour_image": colour_image,
        "depth_image": depth_image,
        "world_from_camera": initial_from_camera
    }


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    batch_mode: bool = args["batch"]
    sequence_dir: str = args["sequence_dir"]

    try:
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_WAIT
        ) as client:
            # Hard-code the camera parameters (for now).
            # FIXME: See https://github.com/ZheC/GTA-IM-Dataset for details on how to obtain the parameters properly.
            calib: CameraParameters = CameraParameters()
            intrinsics: Tuple[float, float, float, float] = (1158.03373708, 1158.03373708, 960.0, 540.0)
            image_size: Tuple[int, int] = (480, 270)
            intrinsics = GeometryUtil.rescale_intrinsics(intrinsics, (1920, 1080), image_size)
            # calib.set("colour", 1920, 1080, 1158.03373708, 1158.03373708, 960.0, 540.0)
            # calib.set("depth", 1920, 1080, 1158.03373708, 1158.03373708, 960.0, 540.0)
            calib.set("colour", *image_size, *intrinsics)
            calib.set("depth", *image_size, *intrinsics)

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            # TODO
            info: List[Dict[str, Any]] = pickle.load(open(os.path.join(sequence_dir, "info_frames.pickle"), "rb"))
            info_npz: np.lib.npyio.NpzFile = np.load(os.path.join(sequence_dir, "info_frames.npz"))

            # Initialise some variables.
            colour_image: Optional[np.ndarray] = None
            depth_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            pause: bool = not batch_mode

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = try_load_frame(frame_idx, sequence_dir, info, info_npz)

                # If the frame was successfully loaded:
                if frame is not None:
                    # Send the frame across to the server.
                    client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                        frame_idx,
                        cv2.resize(frame["colour_image"], image_size),
                        ImageUtil.to_short_depth(cv2.resize(frame["depth_image"], image_size, interpolation=cv2.INTER_NEAREST)),
                        frame["world_from_camera"],
                        msg
                    ))

                    # Increment the frame index.
                    frame_idx += 1

                    # Update the colour and depth images so that they can be shown.
                    colour_image = frame["colour_image"]
                    depth_image = frame["depth_image"]

                # Otherwise, if we're in batch mode, exit.
                elif batch_mode:
                    # noinspection PyProtectedMember
                    os._exit(0)

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow(
                        "GTA-IM Client - Colour Image",
                        cv2.resize(colour_image, (0, 0), fx=0.25, fy=0.25)
                    )
                    cv2.imshow(
                        "GTA-IM Client - Depth Image",
                        cv2.resize(depth_image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST) / 5
                    )

                    if pause:
                        c = cv2.waitKey()
                    else:
                        c = cv2.waitKey(50)

                    if c == ord('b'):
                        pause = False
                    elif c == ord('n'):
                        pause = True
                    elif c == ord('q'):
                        # noinspection PyProtectedMember
                        os._exit(0)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
