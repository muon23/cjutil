from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from recommenders.Recommender import Recommender, Recommendation


class ALSRecommender(Recommender):
    """
    Implicit-feedback recommender based on Alternating Least Squares (ALS).
    """

    def __init__(
            self,
            factors: int = 64,
            regularization: float = 0.01,
            alpha: float = 1.0,
            iterations: int = 15,
            use_gpu: bool = False,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize ALS recommender.

        Args:
            factors: Number of latent factors.
            regularization: L2 regularization strength.
            alpha: Confidence scaling for implicit feedback.
            iterations: ALS optimization iterations.
            use_gpu: Whether to use GPU implementation when available.
            random_state: Optional random seed.
            **kwargs: Additional arguments forwarded to implicit ALS constructor.

        Raises:
            RuntimeError: If `implicit` package is not installed.
            ValueError: If constructor arguments are invalid.
        """
        if factors <= 0:
            raise ValueError("factors must be positive")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as e:
            raise RuntimeError(
                "implicit is required for ALSRecommender. Install it with: pip install implicit"
            ) from e

        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self._model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
            random_state=random_state,
            **kwargs,
        )
        self._is_fitted = False

    @staticmethod
    def _as_csr(matrix: spmatrix) -> csr_matrix:
        # Use scipy constructor for stable runtime + static typing compatibility.
        return matrix if isinstance(matrix, csr_matrix) else csr_matrix(matrix)

    def fit(
            self,
            interactions: spmatrix,
            user_features: spmatrix | None = None,
            item_features: spmatrix | None = None,
            show_progress: bool = False,
            **kwargs: Any,
    ) -> "ALSRecommender":
        """
        Train ALS model on implicit interaction matrix.

        Args:
            interactions: Sparse matrix [num_users, num_items] with interaction strengths.
            user_features: Not used by ALS (accepted for interface compatibility).
            item_features: Not used by ALS (accepted for interface compatibility).
            show_progress: Whether to print ALS training progress.
            **kwargs: Additional fit kwargs forwarded to implicit ALS `fit()`.

        Returns:
            The same ALSRecommender instance.
        """
        _ = user_features, item_features
        user_item = self._as_csr(interactions).astype(np.float32)
        confidence = csr_matrix(user_item * float(self.alpha))
        self._model.fit(confidence, show_progress=show_progress, **kwargs)
        self._is_fitted = True
        return self

    def recommend(
            self,
            user_id: int,
            interactions: spmatrix,
            item_ids: np.ndarray | list[int] | None = None,
            top_k: int = 10,
            user_features: spmatrix | None = None,
            item_features: spmatrix | None = None,
            exclude_known: bool = True,
            recalculate_user: bool = False,
            **kwargs: Any,
    ) -> list[Recommendation]:
        """
        Return top-k recommendations for one user.

        Args:
            user_id: User index.
            interactions: Sparse matrix [num_users, num_items].
            item_ids: Optional candidate item list.
            top_k: Number of recommendations to return.
            user_features: Not used by ALS (accepted for interface compatibility).
            item_features: Not used by ALS (accepted for interface compatibility).
            exclude_known: Whether to exclude already interacted items.
            recalculate_user: Recompute user factors on the fly from input interactions.
            **kwargs: Additional kwargs forwarded to implicit ALS `recommend()`.

        Returns:
            Ranked recommendations in descending score order.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If user_id/top_k are invalid.
        """
        _ = user_features, item_features
        if not self._is_fitted:
            raise RuntimeError("ALSRecommender must be fitted before calling recommend()")
        if top_k <= 0:
            return []

        user_item = self._as_csr(interactions)
        n_users, _ = user_item.shape
        if user_id < 0 or user_id >= n_users:
            raise ValueError(f"user_id {user_id} out of range [0, {n_users - 1}]")
        user_row = user_item[user_id]
        known_items = set(user_row.indices.tolist()) if exclude_known else set()

        items_filter = None
        if item_ids is not None:
            items_filter = np.asarray(item_ids, dtype=np.int32)
            if items_filter.ndim != 1:
                raise ValueError("item_ids must be a 1-D list/array")

        item_ids_out, scores_out = self._model.recommend(
            userid=user_id,
            user_items=user_row,
            N=top_k,
            filter_already_liked_items=exclude_known,
            items=items_filter,
            recalculate_user=recalculate_user,
            **kwargs,
        )

        recs = [
            Recommendation(item_id=int(item_id), score=float(score))
            for item_id, score in zip(item_ids_out, scores_out)
        ]
        if exclude_known:
            recs = [r for r in recs if r.item_id not in known_items]
        return recs[:top_k]
