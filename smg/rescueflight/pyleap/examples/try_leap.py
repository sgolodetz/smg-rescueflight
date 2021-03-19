import numpy as np

from smg.pyleap import leap, LeapController


def main() -> None:
    np.set_printoptions(suppress=True)

    controller: LeapController = LeapController()

    while True:
        frame: leap.Frame = controller.frame()
        if frame.is_valid():
            print(f"Hands: {len(frame.hands())}")

            for i in range(len(frame.hands())):
                hand: leap.Hand = frame.hands()[i]

                hand_name: str = "Hand"
                if hand.is_left():
                    hand_name = "Left hand"
                elif hand.is_right():
                    hand_name = "Right hand"

                print(f"{hand_name} has {len(hand.fingers())} fingers")

                for j in range(len(hand.fingers())):
                    finger: leap.Finger = hand.fingers()[j]
                    fingertip_pos: np.ndarray = controller.from_leap_position(finger.tip_position())
                    finger_dir: np.ndarray = controller.from_leap_direction(finger.direction())
                    print(f"  Finger {finger.type()}: {fingertip_pos}, {finger_dir}")

                    for k in range(4):
                        bone: leap.Bone = finger.bone(leap.EBoneType(k))
                        prev_joint: np.ndarray = controller.from_leap_position(bone.prev_joint())
                        next_joint: np.ndarray = controller.from_leap_position(bone.next_joint())
                        print(f"    Bone {bone.type()}: {prev_joint}, {next_joint}")

            print("===")


if __name__ == "__main__":
    main()
