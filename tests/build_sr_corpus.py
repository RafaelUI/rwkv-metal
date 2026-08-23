"""СЕРБСКИЙ КОРПУС ДЛЯ QLoRA-БЕНЧМАРКА: train/eval из `data/sr.parquet`.

Домен выбран по закону 9: мерить надо там, где база слаба, иначе дельты
трёх плеч утонут в шуме. Сербский -- канарейка этого проекта: он ломался и
от `small=8`, и от `proj=asym_sb6_search` на 2.9B.

Что здесь важно методически:

- **Train и eval разведены ПО СТАТЬЯМ, а не по токенам.** Резать один
  поток на два куска значит оставить в eval хвосты тех же статей, что
  видел train, -- и мерить запоминание, а не перенос.
- **Короткие статьи выброшены.** Половина сербской википедии -- заготовки
  вида «Референце \\n D08»: на них модель учится формату списка, а не
  языку. Порог -- по символам ДО токенизации, чтобы не платить за
  токенизацию мусора.
- **Порядок статей перемешан фиксированным сидом** и записан в шапку: три
  плеча обязаны видеть один и тот же поток в одном и том же порядке,
  иначе сравниваются не базы, а выборки.
- Токены пакуются в непрерывный поток и режутся на окна T+1 (вход и
  сдвинутая цель из одного окна).

    python build_sr_corpus.py [T] [train_окон] [eval_окон]
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import numpy as np
import pyarrow.parquet as pq

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N_EVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MIN_CHARS = 1200
SRC = os.path.expanduser("~/Develop/data/sr.parquet")
OUT = os.path.expanduser("~/Develop/WKV-kvant/sr_qlora_%d.npz" % T)


def main():
    from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer
    tok = WorldTokenizer()

    need_tokens = (N_TRAIN + N_EVAL) * (T + 1)
    print("нужно токенов: %d (train %d окон, eval %d)"
          % (need_tokens, N_TRAIN, N_EVAL), flush=True)

    f = pq.ParquetFile(SRC)
    texts = []
    seen = kept = 0
    for batch in f.iter_batches(batch_size=2000, columns=["text"]):
        for t in batch.column(0).to_pylist():
            seen += 1
            if t is not None and len(t) >= MIN_CHARS:
                texts.append(t)
                kept += 1
        # с запасом: 4 символа на токен -- нижняя оценка для кириллицы,
        # берём вчетверо больше и режем после перемешивания
        if kept * MIN_CHARS > need_tokens * 12:
            break
    print("статей просмотрено %d, оставлено %d (>= %d символов)"
          % (seen, kept, MIN_CHARS), flush=True)

    rs = np.random.RandomState(20260818)
    idx = rs.permutation(len(texts))

    # РАЗВЕДЕНИЕ ПО СТАТЬЯМ: первые статьи -- в eval, остальные -- в train.
    # Так ни один токен eval не встречается в train даже частично.
    n_eval_art = max(1, int(0.06 * len(texts)))
    eval_art = [texts[i] for i in idx[:n_eval_art]]
    train_art = [texts[i] for i in idx[n_eval_art:]]

    def stream(arts, need, name):
        buf = []
        total = 0
        for i, a in enumerate(arts):
            ids = tok.encode(a)
            buf.append(np.asarray(ids, dtype=np.int32))
            buf.append(np.asarray([0], dtype=np.int32))   # разделитель статей
            total += len(ids) + 1
            if total >= need:
                print("  %s: %d токенов из %d статей" % (name, total, i + 1),
                      flush=True)
                break
        else:
            print("  %s: НЕ ХВАТИЛО -- %d токенов из %d статей"
                  % (name, total, len(arts)), flush=True)
        return np.concatenate(buf)[:need]

    ev = stream(eval_art, N_EVAL * (T + 1), "eval")
    tr = stream(train_art, N_TRAIN * (T + 1), "train")

    tr = tr[:len(tr) // (T + 1) * (T + 1)].reshape(-1, T + 1)
    ev = ev[:len(ev) // (T + 1) * (T + 1)].reshape(-1, T + 1)
    np.savez(OUT, train=tr, eval=ev,
             meta=np.array(["sr.parquet", "T=%d" % T, "seed=20260818",
                            "min_chars=%d" % MIN_CHARS,
                            "eval_articles=%d" % n_eval_art]))
    print("\n%s\ntrain %s, eval %s" % (OUT, tr.shape, ev.shape))
    print("контроль: пересечение статей train/eval исключено по построению")
    print("первые 20 токенов train:", tr[0, :20].tolist())
    print("расшифровка:", repr(tok.decode(tr[0, :60].tolist()))[:200])


if __name__ == "__main__":
    main()
