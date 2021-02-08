import cv2
import numpy as np

import smg.pyorbslam3 as pyorbslam3


def main():
    np.set_printoptions(suppress=True)

    with pyorbslam3.RGBDTracker(
        settings_file="settings-kitti.yaml",
        use_viewer=True, voc_file="C:/orbslam3/Vocabulary/ORBvoc.txt", wait_till_ready=True
    ) as tracker:
        for idx in range(100):
            colour_image: np.ndarray = cv2.imread(
                f"D:/datasets/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/orb_slam/frame-{idx:06d}.color.png",
                cv2.IMREAD_UNCHANGED
            )
            depth_image: np.ndarray = cv2.imread(
                f"D:/datasets/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/orb_slam/frame-{idx:06d}.depth.png",
                cv2.IMREAD_UNCHANGED
            )
            print(tracker.estimate_pose(colour_image, depth_image))
            cv2.waitKey(30)


if __name__ == "__main__":
    main()
