import cv2
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.openni import OpenNICamera
from smg.skeletons import Skeleton, SkeletonUtil


def main() -> None:
    # Construct the remote skeleton detector.
    with RemoteSkeletonDetector() as skeleton_detector:
        # Construct the camera.
        with OpenNICamera(mirror_images=True) as camera:
            intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()

            # Repeatedly:
            while True:
                # Get an RGB-D image from the camera.
                colour_image, depth_image = camera.get_images()
                world_from_camera: np.ndarray = np.eye(4)

                # Detect any skeletons in the depth image and 'depopulate' it as necessary.
                depopulated_depth_image: np.ndarray = depth_image.copy()

                skeletons: Optional[List[Skeleton]] = skeleton_detector.detect_skeletons(
                    colour_image, world_from_camera
                )
                if skeletons is not None:
                    start = timer()

                    depopulated_depth_image = SkeletonUtil.depopulate_depth_image(
                        skeletons, depth_image, world_from_camera, intrinsics, debug=True
                    )

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
