import math
import numpy as np
import unittest

from typing import List

from smg.utility import DualNumber, DualQuaternion


class TestDualQuaternion(unittest.TestCase):
    def test_apply(self):
        tolerance: float = 1e-4

        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], -math.pi / 2)
        trans: DualQuaternion = DualQuaternion.from_translation([1, 2, 3])
        dq: DualQuaternion = trans * rot
        self.assertTrue(np.linalg.norm(dq.apply([1, 0, 0]) - np.array([1, 1, 3])) <= tolerance)

    def test_from_axis_angle(self):
        tolerance: float = 1e-4

        axis: np.ndarray = np.array([0, 1, 0])
        dq: DualQuaternion = DualQuaternion.from_axis_angle(axis, math.pi / 4)

        self.assertTrue(np.linalg.norm(dq.get_rotation_vector() - axis * math.pi / 4) <= tolerance)
        self.assertTrue(DualQuaternion.close(dq.get_rotation_part(), dq))
        self.assertTrue(np.linalg.norm(dq.get_translation()) <= tolerance)

        v: np.ndarray = np.array([1, 0, 0])
        w1: np.ndarray = dq.apply(v)

        r: np.ndarray = np.array([0, math.pi / 4, 0])
        w2: np.ndarray = DualQuaternion.from_rotation_vector(r).apply(v)

        self.assertTrue(np.linalg.norm(w2 - w1) <= tolerance)

        expected: DualQuaternion = DualQuaternion(0.707107, 0, 0, 0.707107)

        dq = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        self.assertTrue(DualQuaternion.close(dq, expected))

        dq = DualQuaternion.from_axis_angle([0, 0, -1], -math.pi / 2)
        self.assertTrue(DualQuaternion.close(dq, expected))

        dq = DualQuaternion.from_axis_angle([0, 0, 2], math.pi / 2)
        self.assertTrue(DualQuaternion.close(dq, expected))

        expected = DualQuaternion(0.707107, 0, 0, -0.707107)

        dq = DualQuaternion.from_axis_angle([0, 0, 1], -math.pi / 2)
        self.assertTrue(DualQuaternion.close(dq, expected))

        dq = DualQuaternion.from_axis_angle([0, 0, -1], math.pi / 2)
        self.assertTrue(DualQuaternion.close(dq, expected))

    def test_from_point(self):
        dq: DualQuaternion = DualQuaternion.from_point([3, 4, 5])
        self.assertTrue(DualQuaternion.close(
            dq.conjugate(),
            DualQuaternion(DualNumber(1, 0), DualNumber(0, -3), DualNumber(0, -4), DualNumber(0, -5))
        ))
        self.assertTrue(DualNumber.close(dq.norm(), DualNumber(1, 0)))

    def test_from_se3(self):
        tolerance: float = 1e-4

        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        trans: DualQuaternion = DualQuaternion.from_translation([3, 4, 5])
        v: np.ndarray = np.array([1, 0, 0])

        self.assertTrue(DualQuaternion.close(rot, DualQuaternion(0.707107, 0, 0, 0.707107)))
        self.assertTrue(DualQuaternion.close(
            trans,
            DualQuaternion(1, DualNumber(0, 1.5), DualNumber(0, 2), DualNumber(0, 2.5))
        ))
        self.assertTrue(DualQuaternion.close(
            trans * rot,
            DualQuaternion(
                DualNumber(0.707107, -1.76777),
                DualNumber(0, 2.47487),
                DualNumber(0, 0.353553),
                DualNumber(0.707107, 1.76777)
            )
        ))
        self.assertTrue(np.linalg.norm(rot.apply(v) - np.array([0, 1, 0])) <= tolerance)
        self.assertTrue(np.linalg.norm(trans.apply(v) - np.array([4, 4, 5])) <= tolerance)
        self.assertTrue(np.linalg.norm((trans * rot).apply(v) - np.array([3, 5, 5])) <= tolerance)
        self.assertTrue(np.linalg.norm((rot * trans).apply(v) - np.array([-4, 4, 5])) <= tolerance)

    def test_from_translation(self):
        tolerance: float = 1e-4

        v: np.ndarray = np.array([1, 2, 3])
        t: np.ndarray = np.array([3, 4, 5])
        dq: DualQuaternion = DualQuaternion.from_translation(t)

        self.assertTrue(np.linalg.norm(dq.apply(v) - (v + t)) <= tolerance)
        self.assertTrue(np.linalg.norm(dq.get_rotation_vector()) <= tolerance)
        self.assertTrue(DualQuaternion.close(dq.get_rotation_part(), DualQuaternion.identity()))
        self.assertTrue(np.linalg.norm(dq.get_translation() - t) <= tolerance)
        self.assertTrue(DualQuaternion.close(dq.get_translation_part(), dq))

    def test_linear_blend(self):
        tolerance: float = 1e-4

        p: DualQuaternion = DualQuaternion.from_translation([2, 3, 4])
        q: DualQuaternion = \
            DualQuaternion.from_translation([3, 4, 4]) * DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        v: np.ndarray = np.array([1, 0, 0])

        dqs: List[DualQuaternion] = [p, q]

        self.assertTrue(np.linalg.norm(
            DualQuaternion.linear_blend(dqs, [1.0, 0.0]).apply(v) - np.array([3, 3, 4])
        ) <= tolerance)
        self.assertTrue(np.linalg.norm(
            DualQuaternion.linear_blend(dqs, [0.5, 0.5]).apply(v) - np.array([3.41421, 4, 4])
        ) <= tolerance)
        self.assertTrue(np.linalg.norm(
            DualQuaternion.linear_blend(dqs, [0.0, 1.0]).apply(v) - np.array([3, 5, 4])
        ) <= tolerance)

    def test_pow(self):
        tolerance: float = 1e-4

        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)

        triple_rot: DualQuaternion = rot.pow(3)
        v: np.ndarray = np.array([1, 0, 0])
        self.assertTrue(np.linalg.norm(triple_rot.apply(v) - np.array([0, -1, 0])) <= tolerance)

        trans: DualQuaternion = DualQuaternion.from_translation([3, 4, 5])
        tr: DualQuaternion = trans * rot
        self.assertTrue(np.linalg.norm(tr.pow(3).apply(v) - (tr * tr * tr).apply(v)) <= tolerance)

    def test_sclerp(self):
        tolerance: float = 1e-4

        p: DualQuaternion = DualQuaternion.from_translation([2, 3, 4])
        q: DualQuaternion = \
            DualQuaternion.from_translation([3, 4, 4]) * DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        self.assertTrue(DualQuaternion.close(
            p,
            DualQuaternion(
                DualNumber(1, 0),
                DualNumber(0, 1),
                DualNumber(0, 1.5),
                DualNumber(0, 2)
            )
        ))
        self.assertTrue(DualQuaternion.close(
            q,
            DualQuaternion(
                DualNumber(0.707107, -1.41421),
                DualNumber(0, 2.47487),
                DualNumber(0, 0.353553),
                DualNumber(0.707107, 1.41421)
            )
        ))

        v: np.ndarray = np.array([1, 0, 0])
        self.assertTrue(
            np.linalg.norm(DualQuaternion.sclerp(p, q, 0.0).apply(v) - np.array([3, 3, 4])) <= tolerance
        )
        self.assertTrue(
            np.linalg.norm(DualQuaternion.sclerp(p, q, 0.5).apply(v) - np.array([3.41421, 4, 4])) <= tolerance
        )
        self.assertTrue(
            np.linalg.norm(DualQuaternion.sclerp(p, q, 1.0).apply(v) - np.array([3, 5, 4])) <= tolerance
        )

    def test_screw(self):
        rot: DualQuaternion = DualQuaternion.from_axis_angle([0, 0, 1], math.pi / 2)
        trans: DualQuaternion = DualQuaternion.from_translation([3, 4, 5])
        tr: DualQuaternion = trans * rot
        self.assertTrue(DualQuaternion.close(DualQuaternion.from_screw(tr.to_screw()), tr))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    unittest.main()
