import operator
import unittest

from smg.utility import PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def test_clear(self):
        pq: PriorityQueue[str, float, int] = PriorityQueue[str, float, int](comparator=operator.gt)
        pq.insert("S", 1.0, 23)
        self.assertFalse(pq.empty())
        pq.clear()
        self.assertTrue(pq.empty())

    def test_pop(self):
        pq: PriorityQueue[str, float, int] = PriorityQueue[str, float, int](comparator=operator.gt)
        pq.insert("S", 1.0, 23)
        pq.insert("K", 0.9, 13)
        self.assertTrue(pq.contains("S"))
        self.assertTrue(pq.contains("K"))
        pq.pop()
        self.assertFalse(pq.contains("S"))
        self.assertTrue(pq.contains("K"))

    def test_top(self):
        pq: PriorityQueue[str, float, int] = PriorityQueue[str, float, int](comparator=operator.gt)
        pq.insert("S", 1.0, 23)
        pq.insert("K", 1.1, 13)
        self.assertEqual(pq.top().ident, "K")
        pq.pop()
        self.assertEqual(pq.top().ident, "S")

    def test_update_key(self):
        pq: PriorityQueue[str, float, int] = PriorityQueue[str, float, int](comparator=operator.gt)
        pq.insert("S", 1.0, 23)
        pq.insert("M", 0.9, 7)
        pq.update_key("M", 1.1)
        self.assertEqual(pq.top().ident, "M")
        self.assertAlmostEqual(pq.top().key, 1.1)
        self.assertEqual(pq.top().data, 7)
        pq.pop()
        self.assertEqual(pq.top().ident, "S")
        self.assertAlmostEqual(pq.top().key, 1.0)
        self.assertEqual(pq.top().data, 23)
        pq.pop()
        self.assertTrue(pq.empty())


if __name__ == "__main__":
    unittest.main()
