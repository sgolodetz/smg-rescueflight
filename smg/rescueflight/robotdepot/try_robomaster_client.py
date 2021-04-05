import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from smg.joysticks import FutabaT6K
from smg.robotdepot import RobomasterClient


def main() -> None:
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

    # Construct the Robomaster client.
    with RobomasterClient(("192.168.2.1", 7860)) as client:
        # Repeatedly.
        while True:
            # Process and ignore any pygame events.
            for _ in pygame.event.get():
                pass

            # Update the movement of the robot based on the values output by the Futaba T6K.
            client.set_chassis_fwd(joystick.get_pitch())
            client.set_gimbal_yaw(joystick.get_yaw())


if __name__ == "__main__":
    main()
