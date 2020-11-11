import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.pyorbslam import MonocularTracker
from smg.rotory.drone_factory import DroneFactory


class DepthHandler:
    def __init__(self):
        self.__front_buffer: np.ndarray = np.zeros((256, 320), dtype=float)
        self.__image: Optional[np.ndarray] = None
        self.__pose: Optional[np.ndarray] = None
        self.__reference_image: Optional[np.ndarray] = None
        self.__reference_pose: Optional[np.ndarray] = None
        self.__request_pending: bool = False
        self.__should_terminate: bool = False

        self.__get_lock = threading.Lock()
        self.__request_lock = threading.Lock()

        self.__no_pending_request = threading.Condition(self.__request_lock)
        self.__pending_request = threading.Condition(self.__request_lock)

        depth_thread = threading.Thread(target=self.__process_depth_requests)
        depth_thread.start()

    def get_depth(self):
        with self.__get_lock:
            return self.__front_buffer.copy()

    def request_depth(self, image: np.ndarray, reference_image: np.ndarray, pose: np.ndarray, reference_pose: np.ndarray):
        acquired: bool = self.__request_lock.acquire(blocking=False)
        if acquired:
            self.__image = image
            self.__pose = pose
            self.__reference_image = reference_image
            self.__reference_pose = reference_pose
            self.__request_pending = True
            self.__pending_request.notify()
            self.__request_lock.release()

    def terminate(self):
        self.__should_terminate = True

    def __process_depth_requests(self):
        model_path: str = "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
        # TODO: Use calibrated intrinsics.
        intrinsics: np.ndarray = np.array([
            [921.0, 0.0, 480.0],
            [0.0, 921.0, 360.0],
            [0.0, 0.0, 1.0]
        ])
        depth_estimator: MVDepthEstimator = MVDepthEstimator(model_path, intrinsics)

        with self.__request_lock:
            while not self.__should_terminate:
                while not self.__request_pending:
                    self.__pending_request.wait(0.1)
                    if self.__should_terminate:
                        return

                back_buffer: np.ndarray = depth_estimator.estimate_depth(
                    self.__image, self.__reference_image, self.__pose, self.__reference_pose
                )

                with self.__get_lock:
                    self.__front_buffer = back_buffer

                self.__request_pending = False
                self.__no_pending_request.notify()


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # TODO
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with MonocularTracker(
            settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            reference_image: Optional[np.ndarray] = None
            reference_pose: Optional[np.ndarray] = None

            depth_handler: DepthHandler = DepthHandler()

            _, ax = plt.subplots(1, 2)

            while True:
                image: np.ndarray = drone.get_image()
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord('q'):
                    break

                pose: Optional[np.ndarray] = None
                if tracker.is_ready():
                    pose = tracker.estimate_pose(image)
                if pose is None:
                    continue

                if reference_image is None:
                    reference_image = image
                    reference_pose = pose
                else:
                    depth_handler.request_depth(
                        image, reference_image, pose, reference_pose
                    )

                depth_image: Optional[np.ndarray] = depth_handler.get_depth()
                if depth_image is not None:
                    ax[0].clear()
                    ax[1].clear()
                    ax[0].imshow(image)
                    ax[1].imshow(depth_image)
                    plt.draw()
                    plt.waitforbuttonpress(0.001)

            depth_handler.terminate()


if __name__ == "__main__":
    main()
