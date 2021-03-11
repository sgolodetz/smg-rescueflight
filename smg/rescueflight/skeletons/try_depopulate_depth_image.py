import cv2
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.openni import OpenNICamera
from smg.skeletons import Skeleton, SkeletonUtil


def main() -> None:
    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Construct the remote skeleton detector.
        with RemoteSkeletonDetector() as skeleton_detector:
            intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()

            # Set the camera calibration.
            skeleton_detector.set_calibration(camera.get_colour_size(), intrinsics)

            # Repeatedly:
            while True:
                # Get an RGB-D image from the camera.
                colour_image, depth_image = camera.get_images()
                world_from_camera: np.ndarray = np.eye(4)

                # Detect any skeletons in the depth image and 'depopulate' it as necessary.
                depopulated_depth_image: np.ndarray = depth_image.copy()
                skeletons, people_mask = skeleton_detector.detect_skeletons(colour_image, world_from_camera)

                if skeletons is not None:
                    start = timer()

                    people_mask_from_3d_boxes: np.ndarray = SkeletonUtil.make_people_mask_from_3d_boxes(
                        skeletons, depth_image, world_from_camera, intrinsics
                    )

                    people_mask = np.where(
                        (people_mask != 0) & (people_mask_from_3d_boxes != 0), 255, 0
                    ).astype(np.uint8)

                    depopulated_depth_image = SkeletonUtil.depopulate_depth_image(depth_image, people_mask)

                    end = timer()
                    print(f"Time: {end - start}s")

                # Show the original depth image and the depopulated depth image alongside each other,
                # for comparison purposes. If the user presses 'q', exit.
                cv2.imshow("Depth Image", depth_image / 5)
                cv2.imshow("Depopulated Depth Image", depopulated_depth_image / 5)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break


if __name__ == "__main__":
    main()
