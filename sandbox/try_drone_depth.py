import cv2
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.pyorbslam import MonocularTracker
from smg.rotory.drone_factory import DroneFactory


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
            model_path: str = "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
            # TODO: Use calibrated intrinsics.
            intrinsics: np.ndarray = np.array([
                [921.0, 0.0, 480.0],
                [0.0, 921.0, 360.0],
                [0.0, 0.0, 1.0]
            ])
            depth_estimator: MVDepthEstimator = MVDepthEstimator(model_path, intrinsics)

            reference_image: Optional[np.ndarray] = None
            reference_pose: Optional[np.ndarray] = None

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
                    depth_image: np.ndarray = depth_estimator.estimate_depth(
                        image, reference_image, pose, reference_pose
                    )
                    plt.imshow(depth_image)
                    plt.draw()
                    plt.waitforbuttonpress(0.001)


if __name__ == "__main__":
    main()
