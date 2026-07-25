"""
rwkv_metal.reranker
===================
Cross-encoder реранкер поверх RWKV-7: скор пары (запрос, документ) снимается
не с потокенных активаций, а прямо с рекуррентного состояния базы.

Быстрый старт
-------------
    from rwkv_metal.model import load_pretrained
    from rwkv_metal.tokenizer import WorldTokenizer
    from rwkv_metal.reranker import Reranker, RerankerInference

    base, cfg = load_pretrained("rwkv7-g1d-0.1b.pth")
    model = Reranker(base)                       # база заморожена
    model.load_head("reranker_head.safetensors")

    rr = RerankerInference(model, WorldTokenizer())
    for i, s in rr.rank("как зимуют пчёлы?", docs, top_k=5):
        print(f"{s:+.2f}  {docs[i][:80]}")

Подробности — docs/reranker.md, инференс — docs/inference.md.
"""
from .model import Reranker, RerankerHead, RerankerConfig, resolve_layer_indices
from .data import (
    PairTemplate, RerankSample, DEFAULT_INSTRUCT,
    parse_anchor, load_rows, build_candidates, split_train_eval,
)
from .encode import StateCache, encode_pairs, encode_pairs_direct
from .loss import listwise_loss, bce_loss, mixed_loss
from .train import RerankTrainConfig, train_reranker, evaluate, batch_scores
from .rerank import RerankerInference, DocIndex

__all__ = [
    "Reranker", "RerankerHead", "RerankerConfig", "resolve_layer_indices",
    "PairTemplate", "RerankSample", "DEFAULT_INSTRUCT",
    "parse_anchor", "load_rows", "build_candidates", "split_train_eval",
    "StateCache", "encode_pairs", "encode_pairs_direct",
    "listwise_loss", "bce_loss", "mixed_loss",
    "RerankTrainConfig", "train_reranker", "evaluate", "batch_scores",
    "RerankerInference", "DocIndex",
]
