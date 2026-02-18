from typing import Any

from ml.clustering.ClusterModel import ClusterModel
from ml.clustering.KMeansClusterModel import KMeansClusterModel
from ml.clustering.DBSCANClusterModel import DBSCANClusterModel


def of(model_name: str, **kwargs: Any) -> ClusterModel:
    """
    Factory for clustering model implementations.

    Args:
        model_name: Clustering algorithm name or alias.
        **kwargs: Constructor arguments passed to concrete clustering model.

    Returns:
        Instantiated clustering model wrapper.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    model_name = model_name.lower().strip()
    mapping = {
        "kmeans": KMeansClusterModel,
        "dbscan": DBSCANClusterModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Clustering model {model_name} not supported")
    # Backward-compatible alias for callers that used sklearn naming.
    if model_name == "kmeans" and "k" not in kwargs and "n_clusters" in kwargs:
        kwargs["k"] = kwargs.pop("n_clusters")
    return model_cls(**kwargs)


__all__ = [
    "ClusterModel",
    "KMeansClusterModel",
    "DBSCANClusterModel",
    "of",
]
