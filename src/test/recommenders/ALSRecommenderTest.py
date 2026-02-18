import unittest

import numpy as np
from scipy.sparse import csr_matrix

import recommenders
from recommenders.ALSRecommender import ALSRecommender


class ALSRecommenderTest(unittest.TestCase):
    def setUp(self):
        try:
            import implicit  # noqa: F401
        except ImportError:
            self.skipTest("implicit is not installed")

        self.interactions = csr_matrix(np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ], dtype=np.float32))

    def test_factory_builds_als(self):
        model = recommenders.of("als", factors=8, iterations=3)
        self.assertIsInstance(model, ALSRecommender)

    def test_fit_and_recommend(self):
        model = ALSRecommender(factors=8, iterations=3, random_state=42)
        model.fit(self.interactions, show_progress=False)
        recs = model.recommend(user_id=0, interactions=self.interactions, top_k=3)
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 3)
        self.assertTrue(all(recs[i].score >= recs[i + 1].score for i in range(len(recs) - 1)))

    def test_recommend_excludes_known_items(self):
        model = ALSRecommender(factors=8, iterations=3, random_state=42)
        model.fit(self.interactions, show_progress=False)
        recs = model.recommend(user_id=0, interactions=self.interactions, top_k=5, exclude_known=True)
        known_for_user0 = {0, 3}
        self.assertTrue(all(r.item_id not in known_for_user0 for r in recs))

    def test_recommend_without_fit_raises(self):
        model = ALSRecommender(factors=8, iterations=3, random_state=42)
        with self.assertRaises(RuntimeError):
            model.recommend(user_id=0, interactions=self.interactions, top_k=3)


if __name__ == "__main__":
    unittest.main()
