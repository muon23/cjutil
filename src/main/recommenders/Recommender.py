from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import spmatrix


@dataclass(frozen=True)
class Recommendation:
    """
    One ranked recommendation result.
    """

    item_id: int
    score: float


class Recommender(ABC):
    """
    Base interface for recommender models.
    """

    @abstractmethod
    def fit(
            self,
            interactions: spmatrix,
            user_features: spmatrix | None = None,
            item_features: spmatrix | None = None,
            **kwargs: Any,
    ) -> "Recommender":
        """
        Train recommender on user-item interactions.

        Args:
            interactions: Sparse interaction matrix with shape [num_users, num_items].
            user_features: Optional sparse user-feature matrix.
            item_features: Optional sparse item-feature matrix.
            **kwargs: Backend-specific fitting parameters (e.g., epochs).

        Returns:
            The same recommender instance.

        Raises:
            ValueError: If training inputs are invalid.
        """
        ...

    @abstractmethod
    def recommend(
            self,
            user_id: int,
            interactions: spmatrix,
            item_ids: np.ndarray | list[int] | None = None,
            top_k: int = 10,
            user_features: spmatrix | None = None,
            item_features: spmatrix | None = None,
            exclude_known: bool = True,
            **kwargs: Any,
    ) -> list[Recommendation]:
        """
        Produce top-k recommendations for one user.

        Args:
            user_id: User index in interaction matrix.
            interactions: Sparse interaction matrix with shape [num_users, num_items].
            item_ids: Optional candidate item IDs. If None, uses all items.
            top_k: Number of recommendations to return.
            user_features: Optional sparse user-feature matrix.
            item_features: Optional sparse item-feature matrix.
            exclude_known: Whether to exclude items user already interacted with.
            **kwargs: Backend-specific prediction parameters.

        Returns:
            Ranked recommendation list in descending score order.

        Raises:
            ValueError: If user_id/top_k or matrix shapes are invalid.
        """
        ...

    def recommend_many(
            self,
            user_ids: list[int],
            interactions: spmatrix,
            item_ids: np.ndarray | list[int] | None = None,
            top_k: int = 10,
            user_features: spmatrix | None = None,
            item_features: spmatrix | None = None,
            exclude_known: bool = True,
            **kwargs: Any,
    ) -> dict[int, list[Recommendation]]:
        """
        Default multi-user recommendation implementation.

        Args:
            user_ids: User indices.
            interactions: Sparse interaction matrix with shape [num_users, num_items].
            item_ids: Optional candidate item IDs. If None, uses all items.
            top_k: Number of recommendations per user.
            user_features: Optional sparse user-feature matrix.
            item_features: Optional sparse item-feature matrix.
            exclude_known: Whether to exclude already seen items.
            **kwargs: Backend-specific prediction parameters.

        Returns:
            Mapping from user_id to recommendation list.
        """
        return {
            user_id: self.recommend(
                user_id=user_id,
                interactions=interactions,
                item_ids=item_ids,
                top_k=top_k,
                user_features=user_features,
                item_features=item_features,
                exclude_known=exclude_known,
                **kwargs,
            )
            for user_id in user_ids
        }
