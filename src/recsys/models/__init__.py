"""Classical recommendation models."""

from recsys.models.als import ALSRecommender, SVDSReferenceRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender

__all__ = [
    "ALSRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
    "ItemKNNRecommender",
    "PopularityRecommender",
    "SVDSReferenceRecommender",
]
