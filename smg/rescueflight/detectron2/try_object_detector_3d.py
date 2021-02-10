import cv2
import numpy as np
import open3d as o3d

from detectron2.structures import Instances
from typing import List, Tuple

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera


def main() -> None:
    # Construct the instance segmenter and 3D object detector.
    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
    detector: ObjectDetector3D = ObjectDetector3D(segmenter)

    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Repeatedly:
        while True:
            # Get an RGB-D image from the camera.
            colour_image, depth_image = camera.get_images()

            # Find any 2D instances in the image.
            raw_instances: Instances = segmenter.segment_raw(colour_image)

            # Draw the 2D instances so that they can be shown to the user.
            segmented_image: np.ndarray = segmenter.draw_raw_instances(raw_instances, colour_image)

            # Show the RGB-D image and the 2D instances to the user.
            cv2.imshow("Colour Image", colour_image)
            cv2.imshow("Depth Image", depth_image / 2)
            cv2.imshow("Segmented Image", segmented_image)
            c: int = cv2.waitKey(1)

            # If the user presses 'v', exit the loop.
            if c == ord('v'):
                break

        # Set up the 3D visualisation.
        to_visualise: List[o3d.geometry.Geometry] = []

        intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
        pcd: o3d.geometry.PointCloud = VisualisationUtil.make_rgbd_image_point_cloud(
            colour_image, depth_image, intrinsics
        )
        to_visualise.append(pcd)

        objects: List[ObjectDetector3D.Object3D] = detector.lift_to_3d(
            segmenter.parse_raw_instances(raw_instances), depth_image, np.eye(4), intrinsics
        )

        for obj in objects:
            box: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(*obj.box_3d)
            box.color = (1.0, 0.0, 0.0)
            to_visualise.append(box)

        # Run the 3D visualisation.
        VisualisationUtil.visualise_geometries(to_visualise)


if __name__ == "__main__":
    main()
