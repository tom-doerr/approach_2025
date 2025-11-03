import unittest


class TestTailSplitDisjointFullCover(unittest.TestCase):
    def test_disjoint_and_fullcover_expected_counts(self):
        import vkb.finetune as vf
        # Build synthetic samples: 10 of class 0, 7 of class 1
        samples = [("v0", i, 0) for i in range(10)] + [("v1", i, 1) for i in range(7)]
        tr, va = vf._tail_split(samples, n_classes=2, eval_frac=0.25)
        all_idx = set(range(len(samples)))
        assert set(tr).isdisjoint(set(va))
        assert set(tr) | set(va) == all_idx
        # Expected per-class val sizes
        def expected(c):
            import math
            v = int(round(c * 0.25))
            v = max(1, min(c - 1, v))
            return v
        exp0, exp1 = expected(10), expected(7)
        val0 = sum(1 for i in va if samples[i][2] == 0)
        val1 = sum(1 for i in va if samples[i][2] == 1)
        assert val0 == exp0 and val1 == exp1


if __name__ == "__main__":
    unittest.main()

