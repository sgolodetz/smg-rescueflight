import unittest

from smg.utility import DualNumber


class TestDualNumber(unittest.TestCase):
    def test_conjugate(self):
        dn: DualNumber = DualNumber(2, 3)
        self.assertTrue(DualNumber.close(dn.conjugate(), DualNumber(2, -3)))

    def test_inverse(self):
        dn: DualNumber = DualNumber(2, 3)
        dn_inv: DualNumber = dn.inverse()
        self.assertTrue(DualNumber.close(dn_inv, DualNumber(0.5, -0.75)))
        self.assertTrue(DualNumber.close(dn_inv.inverse(), dn))
        self.assertTrue(DualNumber.close(dn * dn_inv, DualNumber(1)))
        self.assertTrue(DualNumber.close(dn_inv * dn, DualNumber(1)))

    def test_sqrt(self):
        dn: DualNumber = DualNumber(2, 3)
        dn_sqr = dn.sqrt()
        self.assertTrue(DualNumber.close(dn_sqr, DualNumber(1.41421, 1.06066)))
        self.assertTrue(DualNumber.close(dn_sqr * dn_sqr, dn))


if __name__ == "__main__":
    unittest.main()
