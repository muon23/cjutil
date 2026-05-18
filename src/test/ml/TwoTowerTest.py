import importlib.util
import unittest

import numpy as np

from ml.nn import TwoTower

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TwoTowerTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.n = 64
        self.user_X = rng.normal(size=(self.n, 4)).astype(np.float32)
        self.item_X = rng.normal(size=(self.n, 6)).astype(np.float32)
        logits = (self.user_X[:, :2] * self.item_X[:, :2]).sum(axis=1)
        self.y = (logits > 0).astype(np.float32)

    def test_fit_predict_and_encoders(self) -> None:
        model = TwoTower(
            user_dims=(4, 8, 3),
            item_dims=(6, 8, 3),
            epochs=5,
            batch_size=16,
            random_state=42,
        ).fit(self.user_X, self.item_X, self.y)

        scores = model.predict(self.user_X, self.item_X)
        self.assertEqual(self.n, scores.shape[0])

        user_emb = model.encode_user(self.user_X)
        item_emb = model.encode_item(self.item_X)
        self.assertEqual((self.n, 3), user_emb.shape)
        self.assertEqual((self.n, 3), item_emb.shape)

    def test_train_alias(self) -> None:
        model = TwoTower(user_dims=(4, 4), item_dims=(6, 4), epochs=2, batch_size=8)
        trained = model.train(self.user_X, self.item_X, self.y)
        self.assertIs(trained, model)
        self.assertIsNotNone(model.module)

    def test_evaluate(self) -> None:
        model = TwoTower(user_dims=(4, 4), item_dims=(6, 4), epochs=3, batch_size=8).fit(
            self.user_X, self.item_X, self.y
        )
        metrics = model.evaluate(self.user_X, self.item_X, self.y, ["loss", "accuracy"])
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            TwoTower(user_dims=(4, 3), item_dims=(6, 4))
        with self.assertRaises(ValueError):
            TwoTower(user_dims=(4,), item_dims=(6, 4))


if __name__ == "__main__":
    unittest.main()
