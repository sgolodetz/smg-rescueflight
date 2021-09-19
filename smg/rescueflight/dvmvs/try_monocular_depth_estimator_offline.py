import cv2
import numpy as np
import open3d as o3d

from path import Path
from tqdm import tqdm
from typing import List, Optional

from dvmvs.config import Config

from smg.dvmvs import DVMVSMonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.utility import GeometryUtil, ImageUtil


def main() -> None:
    depth_estimator: DVMVSMonocularDepthEstimator = DVMVSMonocularDepthEstimator()

    scene_folder: Path = Path(Config.test_online_scene_path)

    intrinsics: np.ndarray = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses: np.ndarray = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames: List[Path] = sorted((scene_folder / 'images').files("*.png"))

    depth_estimator.set_intrinsics(intrinsics)

    # Make the initial TSDF.
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025,
        sdf_trunc=10 * 0.025,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in tqdm(range(0, len(poses))):
        colour_image: np.ndarray = cv2.imread(image_filenames[i], cv2.IMREAD_COLOR)
        estimated_depth_image: Optional[np.ndarray] = depth_estimator.estimate_depth(colour_image, poses[i])
        if estimated_depth_image is not None:
            cv2.imshow("Colour Image", colour_image)
            cv2.imshow("Estimated Depth Image", estimated_depth_image / 5)
            cv2.waitKey(1)

            # Fuse the frame into the TSDF.
            height, width = colour_image.shape[:2]
            o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, *GeometryUtil.intrinsics_to_tuple(intrinsics)
            )

            ReconstructionUtil.integrate_frame(
                ImageUtil.flip_channels(colour_image), estimated_depth_image, np.linalg.inv(poses[i]),
                o3d_intrinsics, tsdf
            )

    VisualisationUtil.visualise_geometry(ReconstructionUtil.make_mesh(tsdf))


if __name__ == "__main__":
    main()
