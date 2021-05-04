import numpy as np

from smg.pyleap import leap, LeapController


def main() -> None:
    np.set_printoptions(suppress=True)

    controller: LeapController = LeapController()
    controller.enable_gesture(leap.GT_CIRCLE)
    controller.enable_gesture(leap.GT_KEY_TAP)
    controller.enable_gesture(leap.GT_SWIPE)

    while True:
        frame: leap.Frame = controller.frame()
        if frame.is_valid():
            for i in range(len(frame.gestures())):
                gesture: leap.Gesture = frame.gestures()[i]
                if not gesture.is_valid():
                    continue

                if gesture.type() == leap.GT_CIRCLE:
                    circle_gesture: leap.CircleGesture = leap.CircleGesture(gesture)
                    print(
                        f"Circle Gesture ({gesture.id()}, {gesture.state()}): "
                        f"{circle_gesture.centre()}, {circle_gesture.normal()}, {circle_gesture.radius()}, "
                        f"{circle_gesture.progress()}"
                    )
                elif gesture.type() == leap.GT_KEY_TAP:
                    key_tap_gesture: leap.KeyTapGesture = leap.KeyTapGesture(gesture)
                    print(f"Key Tap Gesture ({gesture.id()}, {gesture.state()}): {key_tap_gesture.position()}")
                elif gesture.type() == leap.GT_SWIPE:
                    swipe_gesture: leap.SwipeGesture = leap.SwipeGesture(gesture)
                    print(f"Swipe Gesture ({gesture.id()}, {gesture.state()}): {swipe_gesture.direction()}")


if __name__ == "__main__":
    main()
