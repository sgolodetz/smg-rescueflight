import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from typing import Dict, Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyopencv import CVMat1b
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import *
from smg.rotory.drone_factory import DroneFactory
from smg.utility import GeometryUtil, ImageUtil


def print_se3(se3: SE3f) -> None:
    print()
    for row in range(3):
        print([se3.data(row, col) for col in range(4)])
    print()


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    drone_type: str = args.get("drone_type")

    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with MonocularTracker(
            settings_file=f"settings-tello.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            intrinsics: Optional[Tuple[float, float, float, float]] = drone.get_intrinsics()
            if intrinsics is None:
                raise RuntimeError("Cannot get drone camera intrinsics")

            width, height = drone.get_image_size()
            fx, fy, cx, cy = intrinsics
            depthmap: Depthmap = Depthmap(width, height, fx, cx, fy, cy)

            reference_colour_image: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                # Get an RGB image from the drone.
                colour_image: np.ndarray = drone.get_image()
                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                # Try to estimate the camera pose. If the tracker's not ready, or pose estimation fails, continue.
                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                # Prepare the image and the camera pose to be passed to the depthmap.
                grey_image: np.ndarray = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
                cv_grey_image: CVMat1b = CVMat1b.zeros(*grey_image.shape[:2])
                np.copyto(np.array(cv_grey_image, copy=False), grey_image)

                r: Rotation = Rotation.from_matrix(pose[0:3, 0:3])
                t: np.ndarray = pose[0:3, 3]
                qx, qy, qz, qw = r.as_quat()
                se3: SE3f = SE3f(qw, qx, qy, qz, *t)

                # Pass the image and the camera pose to the depthmap.
                if reference_colour_image is None:
                    reference_colour_image = colour_image
                    depthmap.set_reference_image(cv_grey_image, se3, 0.1, 4.0)
                else:
                    depthmap.update(cv_grey_image, se3)

                # Get the estimated depth image and the convergence map.
                estimated_depth_image: np.ndarray = np.array(depthmap.get_denoised_depthmap())
                convergence_map: np.ndarray = np.array(depthmap.get_convergence_map())

                # Print out the extent to which the depthmap has currently converged.
                print(f"Converged: {depthmap.get_converged_percentage()}%")

                # Visualise the progress towards a suitable depth image. Move on once the user presses a key.
                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(ImageUtil.flip_channels(reference_colour_image))
                ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(ImageUtil.flip_channels(colour_image))

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            # Close any remaining OpenCV windows.
            cv2.destroyAllWindows()

            # Make a point cloud consisting of only those pixels whose depth has converged.
            depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
            pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
                reference_colour_image, estimated_depth_image, depth_mask, (fx, fy, cx, cy)
            )

            # Convert the point cloud to Open3D format.
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

            # Denoise the point cloud (slow).
            pcd = pcd.uniform_down_sample(every_k_points=5)
            pcd, _ = pcd.remove_statistical_outlier(20, 2.0)

            # Visualise the point cloud.
            VisualisationUtil.visualise_geometry(pcd)


if __name__ == "__main__":
    main()
