import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.joysticks import FutabaT6K
from smg.pyorbslam2 import MonocularTracker
from smg.rotory import DroneFactory
from smg.utility import ImageUtil, RGBDSequenceUtil


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save output files"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames that have been obtained from the drone"
    )
    parser.add_argument(
        "--with_tracker", action="store_true", help="whether to use the tracker"
    )
    args: dict = vars(parser.parse_args())

    drone_type: str = args.get("drone_type")

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
    joystick_count = pygame.joystick.get_count()
    joystick_idx = 0
    if joystick_count == 0:
        exit(0)
    elif joystick_count != 1:
        # TODO: Prompt the user for the joystick to use.
        pass

    # Construct and calibrate the Futaba T6K.
    joystick = FutabaT6K(joystick_idx)
    joystick.calibrate()

    # Use the Futaba T6K to control a drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=True, print_control_messages=True, print_navdata_messages=False),
        "tello": dict(print_commands=True, print_responses=True, print_state_messages=False)
    }

    # If requested, construct the tracker.
    tracker: Optional[MonocularTracker] = None
    if args["with_tracker"]:
        tracker = MonocularTracker(
            settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        )

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message) as client:
            calibration_message_needed: bool = True
            frame_idx: int = 0

            # Stop when both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
            while joystick.get_button(0) != 0 or joystick.get_button(1) != 0:
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                        if event.button == 0:
                            drone.takeoff()
                    elif event.type == pygame.JOYBUTTONUP:
                        # If Button 0 on the Futaba T6K is set to its "released" state, land.
                        if event.button == 0:
                            drone.land()

                # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
                drone.move_forward(joystick.get_pitch())
                drone.turn(joystick.get_yaw())

                if joystick.get_button(1) == 0:
                    drone.move_right(0)
                    drone.move_up(joystick.get_roll())
                else:
                    drone.move_right(joystick.get_roll())
                    drone.move_up(0)

                # Get the most recent frame from the drone.
                image: np.ndarray = drone.get_image()
                pose: Optional[np.ndarray] = None

                # If we're using the tracker:
                if tracker is not None:
                    # If the tracker's ready:
                    if tracker.is_ready():
                        # Try to estimate the pose of the camera.
                        inv_pose: np.ndarray = tracker.estimate_pose(image)
                        if inv_pose is not None:
                            pose = np.linalg.inv(inv_pose)
                else:
                    # Otherwise, simply use the identity matrix as a dummy pose.
                    pose = np.eye(4)

                # Send the camera parameters across to the mapping server if we haven't already.
                intrinsics: Tuple[float, float, float, float] = drone.get_intrinsics()
                if calibration_message_needed:
                    height, width = image.shape[:2]
                    client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                        (width, height), (width, height), intrinsics, intrinsics
                    ))
                    calibration_message_needed = False

                # If a pose is available (i.e. unless we were using the tracker and it failed):
                if pose is not None:
                    # Send the frame across to the mapping server.
                    dummy_depth_image: np.ndarray = np.zeros(image.shape[:2], dtype=np.float32)
                    client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                        frame_idx, image, ImageUtil.to_short_depth(dummy_depth_image), pose, msg
                    ))

                    # If an output directory was specified and we're saving frames, save the frame to disk.
                    output_dir: Optional[str] = args.get("output_dir")
                    save_frames: bool = args.get("save_frames")
                    if output_dir is not None and save_frames:
                        RGBDSequenceUtil.save_frame(
                            frame_idx, output_dir, image, dummy_depth_image, pose,
                            colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                        )

                    # Increment the frame index.
                    frame_idx += 1

                # Show the image so that the user can see what's going on (and exit if desired).
                cv2.imshow("Drone Client", image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

    # Shut down pygame cleanly.
    pygame.quit()

    # Forcibly terminate the whole process. This isn't graceful, but ORB-SLAM can sometimes take a long time to
    # shut down, and it's dull to wait for it. For example, if we don't do this, then if we wanted to quit whilst
    # the tracker was still initialising, we'd have to sit around and wait for it to finish, as there's no way to
    # cancel the initialisation.
    # noinspection PyProtectedMember
    os._exit(0)


if __name__ == "__main__":
    main()
