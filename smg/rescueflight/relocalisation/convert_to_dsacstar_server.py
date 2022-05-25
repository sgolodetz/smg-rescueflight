import cv2
import os

from argparse import ArgumentParser

from smg.comms.base import RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.utility import PooledQueue, PoseUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="the directory to which to save the output sequence"
    )
    args: dict = vars(parser.parse_args())

    output_dir: str = args["output_dir"]

    # Determine the paths to the output directories, and make sure they exist.
    output_calibration_dir: str = os.path.join(output_dir, "calibration")
    output_poses_dir: str = os.path.join(output_dir, "poses")
    output_rgb_dir: str = os.path.join(output_dir, "rgb")

    os.makedirs(output_calibration_dir, exist_ok=True)
    os.makedirs(output_poses_dir, exist_ok=True)
    os.makedirs(output_rgb_dir, exist_ok=True)

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.PES_WAIT
    ) as server:
        client_id: int = 0
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Start the server.
        server.start()

        # Repeatedly, until no more frames will be arriving from the client:
        while not server.has_finished(client_id):
            # If the server has a frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the camera parameters from the server.
                fx, fy, cx, cy = server.get_intrinsics(client_id)[0]

                # Get the oldest frame from the server and extract its pose.
                server.get_frame(client_id, receiver)

                # Determine the output filenames.
                output_calibration_filename: str = os.path.join(
                    output_calibration_dir, f"frame-{frame_idx:06d}.calibration.txt"
                )
                output_pose_filename: str = os.path.join(output_poses_dir, f"frame-{frame_idx:06d}.pose.txt")
                output_rgb_filename: str = os.path.join(output_rgb_dir, f"frame-{frame_idx:06d}.color.png")

                # Write the outputs to the relevant files.
                with open(output_calibration_filename, "w") as f:
                    f.write(f"{fx}")
                PoseUtil.save_pose(output_pose_filename, receiver.get_pose())
                # noinspection PyUnresolvedReferences
                cv2.imwrite(output_rgb_filename, receiver.get_rgb_image())

                # Increment the frame index.
                frame_idx += 1


if __name__ == "__main__":
    main()
