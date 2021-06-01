import math
import numpy as np
import unittest

from smg.utility import DualNumber, DualQuaternion, Screw


class TestDualNumber(unittest.TestCase):
    def test_apply(self):
        tolerance: float = 1e-4

        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], -math.pi / 2)
        trans: DualQuaternion = DualQuaternion.from_translation([1, 2, 3])
        dq: DualQuaternion = trans * rot
        self.assertTrue(np.linalg.norm(dq.apply([1, 0, 0]) - np.array([1, 1, 3])) <= tolerance)

    def test_from_axis_angle(self):
        axis: np.ndarray = np.array([0, 1, 0])
        dq: DualQuaternion = DualQuaternion.from_axis_angle(axis, math.pi / 4)
        # TODO

    def test_from_point(self):
        dq: DualQuaternion = DualQuaternion.from_point([3, 4, 5])
        self.assertTrue(DualQuaternion.close(
            dq.conjugate(),
            DualQuaternion(DualNumber(1, 0), DualNumber(0, -3), DualNumber(0, -4), DualNumber(0, -5))
        ))
        self.assertTrue(DualNumber.close(dq.norm(), DualNumber(1, 0)))

    def test_pow(self):
        tolerance: float = 1e-4

        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)

        triple_rot: DualQuaternion = rot.pow(3)
        v: np.ndarray = np.array([1, 0, 0])
        self.assertTrue(np.linalg.norm(triple_rot.apply(v) - np.array([0, -1, 0])) <= tolerance)

        trans: DualQuaternion = DualQuaternion.from_translation([3, 4, 5])
        tr: DualQuaternion = trans * rot
        self.assertTrue(np.linalg.norm(tr.pow(3).apply(v) - (tr * tr * tr).apply(v)) <= tolerance)

    def test_screw(self):
        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        trans: DualQuaternion = DualQuaternion.from_translation([3, 4, 5])
        tr: DualQuaternion = trans * rot
        self.assertTrue(DualQuaternion.close(DualQuaternion.from_screw(tr.to_screw()), tr))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    unittest.main()
