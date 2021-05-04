import numpy as np

from smg.pyleap import leap, LeapController


def main() -> None:
    np.set_printoptions(suppress=True)

    controller: LeapController = LeapController()
    controller.enable_gesture(leap.GT_CIRCLE)
    controller.enable_gesture(leap.GT_SWIPE)

    while True:
        frame: leap.Frame = controller.frame()
        if frame.is_valid():
            for i in range(len(frame.gestures())):
                gesture: leap.Gesture = frame.gestures()[i]
                if gesture.type() == leap.GT_CIRCLE:
                    circle_gesture: leap.CircleGesture = leap.CircleGesture(gesture)
                    print(
                        f"Circle Gesture ({gesture.id()}, {gesture.state()}): "
                        f"{circle_gesture.centre()}, {circle_gesture.normal()}"
                    )
                elif gesture.type() == leap.GT_SWIPE:
                    swipe_gesture: leap.SwipeGesture = leap.SwipeGesture(gesture)
                    print(f"Swipe Gesture ({gesture.id()}, {gesture.state()}): {swipe_gesture.direction()}")


if __name__ == "__main__":
    main()
