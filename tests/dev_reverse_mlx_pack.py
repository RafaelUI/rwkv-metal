import sys
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np, mlx.core as mx

GS = 32
BITS = 6

def quantize_row(vals):
    """vals: python list длиной 32 -> (wq_words[6] as python ints, scale, bias)."""
    w = mx.array([vals], dtype=mx.float32)  # [1,32]
    wq, scales, biases = mx.quantize(w, group_size=GS, bits=BITS)
    return np.array(wq[0]).tolist(), float(scales[0][0]), float(biases[0][0])

# Якоря 0 и 31 фиксируют min/max => фиксируют scale/bias при варьировании
# внутренних позиций 1..30 в узком диапазоне ВНУТРИ [0,31] (не трогая крайности).
base_vals = [0.0] + [15.0]*30 + [31.0]
wq_base, s_base, b_base = quantize_row(base_vals)
print("baseline scale,bias:", s_base, b_base)
print("baseline words:", [f"{w:#010x}" for w in wq_base])

def code_of(v, s=s_base, b=b_base):
    return round((v - b) / s)

print("ожидаемый код для value=15.0:", code_of(15.0))
print("ожидаемый код для value=20.0:", code_of(20.0))

# для каждой внутренней позиции p (1..30) меняем значение на 20.0, смотрим какие биты флипнулись
bit_map = {}
for p in range(1, 31):
    vals = list(base_vals)
    vals[p] = 20.0
    wq_p, s_p, b_p = quantize_row(vals)
    if abs(s_p - s_base) > 1e-6 or abs(b_p - b_base) > 1e-6:
        print(f"  !!! position {p}: scale/bias сдвинулись ({s_p} vs {s_base}) -- якоря выбраны плохо")
        continue
    diff_bits = []
    for wi in range(6):
        xor = wq_p[wi] ^ wq_base[wi]
        if xor:
            for b_ in range(32):
                if xor & (1 << b_):
                    diff_bits.append((wi, b_))
    bit_map[p] = diff_bits
    print(f"position {p:2d} -> изменившиеся биты (word,bit): {diff_bits}")
