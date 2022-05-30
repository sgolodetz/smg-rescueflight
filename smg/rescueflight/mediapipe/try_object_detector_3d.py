import os

import cv2
import numpy as np
import open3d as o3d

from typing import Any, Dict, List, Optional, Tuple

from smg.mediapipe import ObjectDetector3D
from smg.open3d import VisualisationUtil
from smg.utility import SequenceUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # TODO: Comment here.
    intrinsics: Tuple[float, float, float, float] = (
        # 1444.8084716796875, 1444.8084716796875, 962.52960205078125, 726.5919189453125
        191.9573516845703, 191.9573516845703, 128.18414306640625, 96.97889709472656
    )

    # TODO: Comment here.
    # detector: ObjectDetector3D = ObjectDetector3D(image_size=(1920, 1440), intrinsics=intrinsics)
    detector: ObjectDetector3D = ObjectDetector3D(image_size=(256, 192), intrinsics=intrinsics)

    # TODO: Comment here.
    # image: np.ndarray = cv2.imread("frame_01522.jpg")

    # TODO: Comment here.
    # 225 -> 36
    # world_from_camera: np.ndarray = np.array([
    #     [-0.10589541, 0.69620496, -0.70998204, -1.3193058],
    #     [-0.0068612886, 0.7134718, 0.70065033, -1.2010393],
    #     [0.99434847, 0.07906712, -0.07077655, -0.106882855],
    #     [0.0, 0.0, 0.0, 0.99999994]
    # ])

    # 948 -> 154
    # world_from_camera: np.ndarray = np.array([
    #     [-0.92428005, -0.1802313, 0.3364709, -0.00019567888],
    #     [0.11777658, 0.7038291, 0.7005376, -1.2453489],
    #     [-0.3630771, 0.68712145, -0.6293082, -0.35940173],
    #     [0.0, 0.0, 0.0, 0.9999999]
    # ])

    # 1522 -> 248
    # world_from_camera: np.ndarray = np.array([
    #     [0.14105545, -0.5843537, 0.7991395, -0.5030058],
    #     [-0.018675582, 0.8055002, 0.5923012, -1.3135192],
    #     [-0.98982036, -0.098471746, 0.10270709, -1.4744209],
    #     [0.0, 0.0, 0.0, 0.99999994]
    # ])

    fullres_image_filenames: List[str] = sorted([f for f in os.listdir("C:/iphonescans/2022_05_17_14_37_57") if f.endswith(".jpg")])

    frame_idx: int = 36
    frame: Optional[Dict[str, Any]] = SequenceUtil.try_load_rgbd_frame(
        frame_idx, "C:/iphonescans/2022_05_17_14_37_57-aligned"
    )
    image: np.ndarray = cv2.imread(os.path.join("C:/iphonescans/2022_05_17_14_37_57", fullres_image_filenames[frame_idx]))
    world_from_camera: np.ndarray = frame["world_from_camera"]

    # Set up the 3D visualisation.
    to_visualise: List[o3d.geometry.Geometry] = []

    # TODO: Comment here.
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("C:/iphonescans/2022_05_17_14_37_57-aligned/mesh.ply")
    to_visualise.append(mesh)

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
    objects: List[ObjectDetector3D.Object3D] = detector.detect_objects(image, world_from_camera)
    for obj in objects:
        print(obj.landmarks_3d)
        for i, landmark_3d in enumerate(obj.landmarks_3d):
            to_visualise.append(VisualisationUtil.make_sphere(landmark_3d, 0.01, colour=colours[i]))

        to_visualise.append(VisualisationUtil.make_obb(obj.landmarks_3d[1:], colour=(0, 1, 0)))

    # TODO: Comment here.
    VisualisationUtil.visualise_geometries(to_visualise, initial_pose=np.linalg.inv(world_from_camera))


if __name__ == "__main__":
    main()
