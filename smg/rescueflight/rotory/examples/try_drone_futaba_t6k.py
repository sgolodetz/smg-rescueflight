import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from typing import Dict, List

from smg.rotory import DroneFactory
from smg.rotory.controllers import DroneController, FutabaT6KDroneController


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "simulated", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Construct the drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=True, print_control_messages=True, print_navdata_messages=False),
        "simulated": dict(),
        "tello": dict(print_commands=True, print_responses=True, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        # Construct the drone controller.
        drone_controller: DroneController = FutabaT6KDroneController(drone=drone)

        # Stop when the drone controller asks us to quit.
        while not drone_controller.should_quit():
            # Record any PyGame events for later use by the drone controller.
            events: List[pygame.event.Event] = []
            for event in pygame.event.get():
                events.append(event)

            # Get the most recent image from the drone and show it.
            drone_image: np.ndarray = drone.get_image()
            cv2.imshow("Image", drone_image)
            cv2.waitKey(1)

            # Allow the user to control the drone.
            drone_controller.iterate(events=events, image=drone_image, intrinsics=drone.get_intrinsics())

            # FIXME: Delete these before merging.
            # from smg.rotory.drones import SimulatedDrone
            # sim_drone: SimulatedDrone = drone
            # _, drone_w_t_c, _ = sim_drone.get_image_and_poses()
            # print(drone_w_t_c)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
