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
    image: np.ndarray = cv2.imread("frame_00225.jpg")

    # TODO: Comment here.
    # 225 -> 36
    world_from_camera: np.ndarray = np.array([
        [-0.10589541, 0.69620496, -0.70998204, -1.3193058],
        [-0.0068612886, 0.7134718, 0.70065033, -1.2010393],
        [0.99434847, 0.07906712, -0.07077655, -0.106882855],
        [0.0, 0.0, 0.0, 0.99999994]
    ])

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

        print(np.linalg.norm(obj.landmarks_3d[2] - obj.landmarks_3d[1]))
        print(np.linalg.norm(obj.landmarks_3d[4] - obj.landmarks_3d[3]))
        print(np.linalg.norm(obj.landmarks_3d[3] - obj.landmarks_3d[1]))
        print(np.linalg.norm(obj.landmarks_3d[4] - obj.landmarks_3d[2]))
        print(np.linalg.norm(obj.landmarks_3d[5] - obj.landmarks_3d[1]))
        print(np.linalg.norm(obj.landmarks_3d[6] - obj.landmarks_3d[2]))
        # for i in range(len(obj.landmarks_3d)):
        #     for j in range(len(obj.landmarks_3d)):
        #         print(i, j, np.linalg.norm(obj.landmarks_3d[i] - obj.landmarks_3d[j]))

    # TODO: Comment here.
    VisualisationUtil.visualise_geometries(to_visualise, initial_pose=np.linalg.inv(world_from_camera))


if __name__ == "__main__":
    main()
