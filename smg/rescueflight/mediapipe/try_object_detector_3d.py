import cv2
import numpy as np
import open3d as o3d

from typing import List, Tuple

from smg.mediapipe import ObjectDetector3D
from smg.open3d import VisualisationUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # TODO: Comment here.
    intrinsics: Tuple[float, float, float, float] = (
        1444.8084716796875, 1444.8084716796875, 962.52960205078125, 726.5919189453125
    )

    # TODO: Comment here.
    detector: ObjectDetector3D = ObjectDetector3D(image_size=(1920, 1440), intrinsics=intrinsics)

    # TODO: Comment here.
    image: np.ndarray = cv2.imread("frame_01522.jpg")

    # TODO: Comment here.
    # 948 -> 154
    # world_from_camera: np.ndarray = np.array([
    #     [-0.92428005, -0.1802313, 0.3364709, -0.00019567888],
    #     [0.11777658, 0.7038291, 0.7005376, -1.2453489],
    #     [-0.3630771, 0.68712145, -0.6293082, -0.35940173],
    #     [0.0, 0.0, 0.0, 0.9999999]
    # ])

    # 1522 -> 248
    world_from_camera: np.ndarray = np.array([
        [0.14105545, -0.5843537, 0.7991395, -0.5030058],
        [-0.018675582, 0.8055002, 0.5923012, -1.3135192],
        [-0.98982036, -0.098471746, 0.10270709, -1.4744209],
        [0.0, 0.0, 0.0, 0.99999994]
    ])

    # Set up the 3D visualisation.
    to_visualise: List[o3d.geometry.Geometry] = []

    # TODO: Comment here.
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("C:/iphonescans/2022_05_17_14_37_57-aligned/mesh.ply")
    to_visualise.append(mesh)

    # TODO: Comment here.
    objects: List[ObjectDetector3D.Object3D] = detector.detect_objects(image, world_from_camera)
    for obj in objects:
        print(obj.landmarks_3d)
        for landmark_3d in obj.landmarks_3d:
            to_visualise.append(VisualisationUtil.make_sphere(landmark_3d, 0.01, colour=(1, 0, 0)))

    # TODO: Comment here.
    VisualisationUtil.visualise_geometries(to_visualise, initial_pose=np.linalg.inv(world_from_camera))


if __name__ == "__main__":
    main()
