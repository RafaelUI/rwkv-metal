"""
rwkv_metal.tokenizer
====================
RWKV World tokenizer (65536-token vocab) used by official RWKV-7 World models.

    from rwkv_metal.tokenizer import WorldTokenizer
    tok = WorldTokenizer()                 # bundled vocab
    ids = tok.encode("Hello world")
    text = tok.decode(ids)

Note: the from-scratch training track uses its own tokenizer (e.g. a 21k/32k
BPE); the World tokenizer is only for official x070 weights. They are NOT
interchangeable.
"""
from .world_tokenizer import WorldTokenizer, TRIE

__all__ = ["WorldTokenizer", "TRIE"]
