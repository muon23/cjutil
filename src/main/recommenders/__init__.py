from recommenders.ALSRecommender import ALSRecommender
from recommenders.Recommender import Recommender, Recommendation


def of(model_name: str, **kwargs) -> Recommender:
    """
    Factory for recommender implementations.

    Args:
        model_name: Recommender model name or alias.
        **kwargs: Constructor arguments for concrete recommender.

    Returns:
        Recommender implementation instance.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    model_name = model_name.lower().strip()
    if model_name in {"als", "implicit_als"}:
        return ALSRecommender(**kwargs)
    raise RuntimeError(f"Recommender model {model_name} not supported")


__all__ = [
    "Recommender",
    "Recommendation",
    "ALSRecommender",
    "of",
]
