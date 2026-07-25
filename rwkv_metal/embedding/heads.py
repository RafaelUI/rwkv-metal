"""
Trainable pooling head on top of a base RWKV-7 model's hidden state.

Kept as a SEPARATE module from the base model on purpose: `add_lora()` /
`quantize_base_model()` call `model.freeze()` on the base tree, which would
silently freeze this head too if it lived inside that tree. `EmbeddingHead`
stays a sibling of the base (see `EmbeddingModel` in `train.py`), so it stays
trainable regardless of what happens to the base underneath it (full-FT,
frozen layers, or LoRA/QLoRA-wrapped).
"""
import mlx.core as mx
import mlx.nn as nn


class EmbeddingHead(nn.Module):
    """Zero-init residual head: identity at step 0 (doesn't disturb the
    base's pretrained geometry before any training happens), learns a
    correction on top of it during contrastive fine-tuning.

    Same shape as EmbeddingRWKV's NonlinearHead, single-task (no [CLS]/[STS]/
    [RETR] split) -- add more heads later (one per task) if needed.
    """

    def __init__(self, dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.fc2.weight = mx.zeros_like(self.fc2.weight)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc2(nn.relu(self.fc1(x)))
        return self.norm(h + x)
