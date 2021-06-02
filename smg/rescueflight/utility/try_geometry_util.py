import math
import unittest

from smg.utility import DualQuaternion, GeometryUtil


class TestGeometryUtil(unittest.TestCase):
    def test_poses_are_similar(self):
        rotation_threshold: float = 20 * math.pi / 180
        translation_threshold: float = 0.05

        r1: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 0.0)
        r2: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 19 * math.pi / 180)
        r3: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], 21 * math.pi / 180)

        self.assertTrue(GeometryUtil.poses_are_similar(
            r1.to_rigid_matrix(), r2.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.poses_are_similar(
            r1.to_rigid_matrix(), r3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertTrue(GeometryUtil.poses_are_similar(
            r2.to_rigid_matrix(), r3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))

        t1: DualQuaternion = DualQuaternion.from_translation([0, 0, 0])
        t2: DualQuaternion = DualQuaternion.from_translation([0.04, 0, 0])
        t3: DualQuaternion = DualQuaternion.from_translation([0.06, 0, 0])

        self.assertTrue(GeometryUtil.poses_are_similar(
            t1.to_rigid_matrix(), t2.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.poses_are_similar(
            t1.to_rigid_matrix(), t3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertTrue(GeometryUtil.poses_are_similar(
            t2.to_rigid_matrix(), t3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))

        t1r1: DualQuaternion = t1 * r1
        t1r3: DualQuaternion = t1 * r3
        t2r2: DualQuaternion = t2 * r2
        t3r1: DualQuaternion = t3 * r1
        t3r3: DualQuaternion = t3 * r3

        self.assertTrue(GeometryUtil.poses_are_similar(
            t1r1.to_rigid_matrix(), t2r2.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.poses_are_similar(
            t1r1.to_rigid_matrix(), t1r3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.poses_are_similar(
            t1r1.to_rigid_matrix(), t3r1.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))
        self.assertFalse(GeometryUtil.poses_are_similar(
            t1r1.to_rigid_matrix(), t3r3.to_rigid_matrix(),
            rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
        ))


if __name__ == "__main__":
    unittest.main()
