import cv2
import numpy as np

from smg.rotory.drones import Tello


def main():
    # Connect to the Tello TT.
    with Tello(print_commands=True, print_responses=True, print_state_messages=False) as drone:
        # Activate the drone's motors to keep it cool.
        drone.activate_motors()

        # Repeatedly:
        while True:
            # Render a simple image onto the drone's LED panel.
            drone.send_custom_command(
                "EXT mled g 0000000000pppp0000prrp000prrrrp00prrrrp000rrrr000bbrrbb0bbbrrbbbbbbbbbbbbbbbbbbb"
            )

            # Get the most recent image received from the drone, and show it. If the user presses 'q', exit.
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord('q'):
                break

        # Deactivate the drone's motors again.
        drone.deactivate_motors()


if __name__ == "__main__":
    main()
