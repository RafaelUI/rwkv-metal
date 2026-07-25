from .embed import embed_texts, Embedder, cosine_similarity_matrix
from .heads import EmbeddingHead
from .dataset import (
    load_pairs_jsonl,
    load_triplets_jsonl,
    encode_batch,
    PairBatcher,
    TripletBatcher,
    ClassificationBatcher,
    parse_classification_candidates,
)
from .loss import info_nce_loss, triplet_pool_loss, zero_shot_classification_loss
from .tasks import (
    pair_loss, retrieval_loss, sts_loss, classification_loss,
    RETRIEVAL_GC, STS_GC,
)
from .gradcache import GradCacheSpec, gradcache_value_and_grad
from .train import EmbeddingModel, EmbedTrainConfig, finetune_embedding, save_embedding_model
from .eval import evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification

__all__ = [
    "embed_texts", "Embedder", "cosine_similarity_matrix",
    "EmbeddingHead",
    "load_pairs_jsonl", "load_triplets_jsonl", "encode_batch",
    "PairBatcher", "TripletBatcher", "ClassificationBatcher",
    "parse_classification_candidates",
    "info_nce_loss", "triplet_pool_loss", "zero_shot_classification_loss",
    "pair_loss", "retrieval_loss", "sts_loss", "classification_loss",
    "RETRIEVAL_GC", "STS_GC", "GradCacheSpec", "gradcache_value_and_grad",
    "EmbeddingModel", "EmbedTrainConfig", "finetune_embedding", "save_embedding_model",
    "evaluate_retrieval", "evaluate_sts_pairwise", "evaluate_classification",
]
