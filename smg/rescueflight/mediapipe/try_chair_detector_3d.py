import os

import cv2
import numpy as np
import open3d as o3d

from typing import Any, Dict, List, Optional, Tuple

from smg.mediapipe import ChairDetector3D
from smg.open3d import VisualisationUtil
from smg.utility import SequenceUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # TODO: Comment here.
    intrinsics: Tuple[float, float, float, float] = (
        1444.8084716796875, 1444.8084716796875, 962.52960205078125, 726.5919189453125
        # 191.9573516845703, 191.9573516845703, 128.18414306640625, 96.97889709472656
    )

    # TODO: Comment here.
    detector: ChairDetector3D = ChairDetector3D(debug=True, image_size=(1920, 1440), intrinsics=intrinsics)
    # detector: ChairDetector3D = ChairDetector3D(debug=True, image_size=(256, 192), intrinsics=intrinsics)

    # Set up the 3D visualisation.
    to_visualise: List[o3d.geometry.Geometry] = []

    # TODO: Comment here.
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("C:/iphonescans/2022_05_17_14_37_57-aligned/mesh.ply")
    to_visualise.append(mesh)

    fullres_image_filenames: List[str] = sorted([
        f for f in os.listdir("C:/iphonescans/2022_05_17_14_37_57") if f.endswith(".jpg")
    ])

    frame_idx: int = 0
    world_from_camera: np.ndarray = np.eye(4)
    while True:
        frame: Optional[Dict[str, Any]] = SequenceUtil.try_load_rgbd_frame(
            frame_idx, "C:/iphonescans/2022_05_17_14_37_57-aligned"
        )

        if frame is None:
            break

        image: np.ndarray = cv2.imread(os.path.join("C:/iphonescans/2022_05_17_14_37_57", fullres_image_filenames[frame_idx]))
        world_from_camera = frame["world_from_camera"]

        print(frame_idx)
        frame_idx += 1

        # TODO: Comment here.
        colours: List[Tuple[float, float, float]] = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 0.5, 0),
            (1, 0, 0.5),
            (0, 0.5, 1)
        ]
        detected_chairs: List[ChairDetector3D.Chair] = detector.detect_chairs(image, world_from_camera)
        for chair in detected_chairs:
            for i, landmark_3d in enumerate(chair.landmarks_3d):
                to_visualise.append(VisualisationUtil.make_sphere(landmark_3d, 0.01, colour=colours[i]))

            to_visualise.append(VisualisationUtil.make_obb(chair.landmarks_3d[1:], colour=(0, 1, 0)))

    # TODO: Comment here.
    VisualisationUtil.visualise_geometries(to_visualise)  # , initial_pose=np.linalg.inv(world_from_camera))


if __name__ == "__main__":
    main()
