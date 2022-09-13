import cv2
import numpy as np

from smg.rotory.drones import Tello


def main():
    # Connect to the Tello.
    with Tello(print_commands=True, print_responses=True, print_state_messages=False) as tello:
        # Activate the Tello's motors to keep it cool.
        tello.activate_motors()

        # Repeatedly:
        while True:
            # TODO: Comment here.
            tello.send_custom_command(
                "EXT mled g 0000000000pppp0000prrp000prrrrp00prrrrp000rrrr000bbrrbb0bbbrrbbbbbbbbbbbbbbbbbbb"
            )

            # TODO: Comment here.
            image: np.ndarray = tello.get_image()
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord('q'):
                break

        # Deactivate the Tello's motors again.
        tello.deactivate_motors()


if __name__ == "__main__":
    main()
