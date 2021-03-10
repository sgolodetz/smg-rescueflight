import cv2
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.mvdepthnet import MonocularDepthEstimator
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.skeletons import Skeleton, SkeletonUtil
from smg.utility import GeometryUtil


def main() -> None:
    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Construct the tracker.
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=False,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Construct the remote skeleton detector.
            with RemoteSkeletonDetector() as skeleton_detector:
                # Construct the depth estimator.
                intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
                depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
                    "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True,
                    max_consistent_depth_diff=np.inf
                ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Repeatedly:
                while True:
                    # Get the colour and depth images from the camera, and show them. If the user presses 'q', exit.
                    colour_image, depth_image = camera.get_images()
                    cv2.imshow("Colour Image", colour_image)
                    cv2.imshow("Depth Image", depth_image / 5)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break

                    # If the tracker's not yet ready, or the pose can't be estimated for this frame, continue.
                    if not tracker.is_ready():
                        continue

                    tracker_c_t_w: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                    if tracker_c_t_w is None:
                        continue

                    tracker_w_t_c: np.ndarray = np.linalg.inv(tracker_c_t_w)

                    # Estimate the depth image.
                    estimated_depth_image: Optional[np.ndarray] = depth_estimator.estimate_depth(
                        colour_image, tracker_w_t_c
                    )

                    # If a depth image was successfully estimated:
                    if estimated_depth_image is not None:
                        # Detect any skeletons in the estimated depth image and 'depopulate' it as necessary.
                        depopulated_depth_image: np.ndarray = estimated_depth_image.copy()
                        skeletons, people_mask = skeleton_detector.detect_skeletons(colour_image, tracker_w_t_c)

                        if skeletons is not None:
                            start = timer()

                            depopulated_depth_image = SkeletonUtil.depopulate_depth_image(
                                skeletons, estimated_depth_image, tracker_w_t_c, intrinsics, debug=True
                            )

                            end = timer()
                            print(f"Time: {end - start}s")

                        # Show the estimated depth image and the depopulated depth image alongside each other,
                        # for comparison purposes. If the user presses 'q', exit.
                        cv2.imshow("Estimated Depth Image", estimated_depth_image / 5)
                        cv2.imshow("Depopulated Depth Image", depopulated_depth_image / 5)
                        c: int = cv2.waitKey(1)
                        if c == ord('q'):
                            break


if __name__ == "__main__":
    main()
