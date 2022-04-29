import math
import numpy as np
import unittest

from typing import List

from smg.utility import DualQuaternion, GeometryUtil


class TestGeometryUtil(unittest.TestCase):
    def test_distance_to_line_segment(self):
        self.assertAlmostEqual(GeometryUtil.distance_to_line_segment(
            [0, 0, 0], [1, 1, 0], [1, 2, 0]
        ), np.sqrt(2))
        self.assertAlmostEqual(GeometryUtil.distance_to_line_segment(
            [0, 0, 0], [1, 0, 0], [0, 1, 0]
        ), 1 / np.sqrt(2))

    def test_find_closest_point_on_half_ray(self):
        tolerance: float = 1e-4
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_half_ray(
            [0, 0, 0], [1, -1, 0], [0, 1, 0]
        ) - np.array([1, 0, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_half_ray(
            [0, 0, 0], [1, 1, 0], [0, 1, 0]
        ) - np.array([1, 1, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_half_ray(
            [0, 0, 0], [2, -1, 0], [-1, 1, 0]
        ) - np.array([0.5, 0.5, 0])) <= tolerance)

    def test_find_closest_point_on_line_segment(self):
        tolerance: float = 1e-4
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0, 0, 0], [1, 1, 0], [1, 1, 0]
        ) - np.array([1, 1, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0, 0, 0], [1, 1, 0], [1, 2, 0]
        ) - np.array([1, 1, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0, 0, 0], [1, -2, 0], [1, -1, 0]
        ) - np.array([1, -1, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0, 0, 0], [1, -1, 0], [1, 1, 0]
        ) - np.array([1, 0, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0, 0, 0], [1, 0, 0], [0, 1, 0]
        ) - np.array([0.5, 0.5, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0.9, 0, 0], [1, 0, 0], [0, 1, 0]
        ) - np.array([0.95, 0.05, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(GeometryUtil.find_closest_point_on_line_segment(
            [0.9, -0.1, 0], [1, 0, 0], [0, 1, 0]
        ) - np.array([1, 0, 0])) <= tolerance)

    def test_find_largest_cluster(self):
        # Generate increasingly large clusters of transforms that rotate around the z axis by different angles.
        up: np.ndarray = np.array([0, 0, 1])
        transforms: List[np.ndarray] = []

        for i in range(4):
            for j in range(-i, i + 1):
                angle: float = i * math.pi / 2 + j * math.pi / 180
                transforms.append(DualQuaternion.from_axis_angle(up, angle).to_rigid_matrix())

        # Find the largest cluster, and check that the transforms are those that rotate by the angles close to 3*PI/2.
        rotation_threshold: float = 20 * math.pi / 180
        translation_threshold: float = 0.05
        cluster: List[int] = GeometryUtil.find_largest_cluster(
            transforms, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        )
        self.assertTrue(cluster == list(range(1 + 3 + 5, 1 + 3 + 5 + 7)))

        # Check that blending the transforms in the cluster gives a rotation by 3*PI/2.
        refined_transform: np.ndarray = GeometryUtil.blend_rigid_transforms([transforms[i] for i in cluster])
        self.assertTrue(DualQuaternion.close(
            DualQuaternion.from_rigid_matrix(refined_transform),
            DualQuaternion.from_axis_angle(up, 3 * math.pi / 2)
        ))

    def test_transforms_are_similar(self):
        rotation_threshold: float = 20 * math.pi / 180
        translation_threshold: float = 0.05

        r1: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 0.0)
        r2: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 19 * math.pi / 180)
        r3: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 21 * math.pi / 180)

        self.assertTrue(GeometryUtil.transforms_are_similar(
            r1, r2, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.transforms_are_similar(
            r1, r3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertTrue(GeometryUtil.transforms_are_similar(
            r2, r3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))

        t1: DualQuaternion = DualQuaternion.from_translation([0, 0, 0])
        t2: DualQuaternion = DualQuaternion.from_translation([0.04, 0, 0])
        t3: DualQuaternion = DualQuaternion.from_translation([0.06, 0, 0])

        self.assertTrue(GeometryUtil.transforms_are_similar(
            t1, t2, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.transforms_are_similar(
            t1, t3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertTrue(GeometryUtil.transforms_are_similar(
            t2, t3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))

        t1r1: DualQuaternion = t1 * r1
        t1r3: DualQuaternion = t1 * r3
        t2r2: DualQuaternion = t2 * r2
        t3r1: DualQuaternion = t3 * r1
        t3r3: DualQuaternion = t3 * r3

        self.assertTrue(GeometryUtil.transforms_are_similar(
            t1r1, t2r2, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.transforms_are_similar(
            t1r1, t1r3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.transforms_are_similar(
            t1r1, t3r1, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.transforms_are_similar(
            t1r1, t3r3, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    unittest.main()
