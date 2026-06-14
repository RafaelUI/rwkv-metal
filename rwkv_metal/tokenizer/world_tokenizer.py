"""RWKV World tokenizer (TRIE), чистый Python, без torch."""

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    def __init__(self, front=None, ch=None):
        self.ch = ch; self.to = [None] * 256; self.values = set(); self.front = front
    def add(self, key, idx=0, val=None):
        if idx == len(key):
            if val is None: val = key
            self.values.add(val); return self
        ch = key[idx]
        if self.to[ch] is None: self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx + 1, val)
    def find_longest(self, key, idx=0):
        u = self; ch = key[idx]; ret = None
        while u.to[ch] is not None:
            u = u.to[ch]; idx += 1
            if u.values: ret = idx, u, u.values
            if idx == len(key): break
            ch = key[idx]
        return ret

class WorldTokenizer:
    """RWKV World tokenizer (65536-token vocab), pure Python, torch-free.

    file_name: path to rwkv_vocab_v20230424.txt. Defaults to the copy bundled
    with the package, so WorldTokenizer() just works.
    """
    def __init__(self, file_name=None):
        if file_name is None:
            import os
            file_name = os.path.join(os.path.dirname(__file__),
                                     "rwkv_vocab_v20230424.txt")
        self.idx2token = {}
        with open(file_name, "r", encoding="utf-8") as f:
            for l in f:
                idx = int(l[:l.index(' ')])
                x = eval(l[l.index(' '):l.rindex(' ')])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes) and len(x) == int(l[l.rindex(' '):])
                self.idx2token[idx] = x
        self.token2idx = {v: k for k, v in self.idx2token.items()}
        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=(t, i))
    def encode(self, s):
        src = s.encode("utf-8"); idx = 0; toks = []
        while idx < len(src):
            ret = self.root.find_longest(src, idx); idx = ret[0]
            _, tok = next(iter(ret[2])); toks.append(tok)
        return toks
    def decode(self, tokens):
        return b''.join(self.idx2token[i] for i in tokens).decode('utf-8', errors='replace')
