import cv2
import numpy as np
import open3d as o3d

from path import Path
from tqdm import tqdm
from typing import List, Optional

from dvmvs.config import Config

from smg.dvmvs import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.utility import ImageUtil


def main() -> None:
    Config.test_visualize = False

    depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator()

    scene_folder: Path = Path(Config.test_online_scene_path)

    intrinsics: np.ndarray = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses: np.ndarray = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames: List[Path] = sorted((scene_folder / 'images').files("*.png"))

    depth_estimator.set_input_intrinsics(intrinsics)

    # Make the initial TSDF.
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025,
        sdf_trunc=5 * 0.025,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in tqdm(range(0, len(poses))):
        colour_image: np.ndarray = cv2.imread(image_filenames[i], cv2.IMREAD_COLOR)
        output_depth_image, output_colour_image = depth_estimator.estimate_depth(colour_image, poses[i])
        if output_depth_image is not None:
            cv2.imshow("Colour Image", output_colour_image)
            cv2.imshow("Estimated Depth Image", output_depth_image / 5)
            output_intrinsics: Optional[np.ndarray] = depth_estimator.get_output_intrinsics()
            print()
            print(output_intrinsics)

            # postprocessed_depth_image: Optional[np.ndarray] = MonocularDepthEstimator.postprocess_depth_image(
            #     output_depth_image.astype(np.float32)
            # )
            # if postprocessed_depth_image is None:
            #     cv2.waitKey(1)
            #     continue

            prediction_height, prediction_width = output_depth_image.shape[:2]
            edge_pixel_amount = 10
            edge_mask = np.zeros((prediction_height, prediction_width), dtype=bool)
            edge_mask[0:edge_pixel_amount, :] = True
            edge_mask[prediction_height - edge_pixel_amount: prediction_height, :] = True
            edge_mask[:, 0:edge_pixel_amount] = True
            edge_mask[:, prediction_width - edge_pixel_amount: prediction_width] = True

            # black_mask = np.mean(output_depth_image.astype(float), axis=-1) < 10.0
            # combined_mask = np.logical_and(black_mask, edge_mask)
            postprocessed_depth_image = np.where(edge_mask == 0, output_depth_image, 0.0)
            postprocessed_depth_image = np.where(postprocessed_depth_image <= 5.0, postprocessed_depth_image, 0.0)

            cv2.imshow("Postprocessed Depth Image", postprocessed_depth_image / 5)
            cv2.waitKey(1)

            # Fuse the frame into the TSDF.
            o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                320, 256, output_intrinsics[0][0], output_intrinsics[1][1],
                output_intrinsics[0][2], output_intrinsics[1][2]
            )

            ReconstructionUtil.integrate_frame(
                ImageUtil.flip_channels(output_colour_image), postprocessed_depth_image, np.linalg.inv(poses[i]),
                o3d_intrinsics, tsdf
            )

            # VisualisationUtil.visualise_rgbd_image(
            #     output_colour_image, output_depth_image, GeometryUtil.intrinsics_to_tuple(output_intrinsics)
            # )

    VisualisationUtil.visualise_geometry(ReconstructionUtil.make_mesh(tsdf))


if __name__ == "__main__":
    main()
