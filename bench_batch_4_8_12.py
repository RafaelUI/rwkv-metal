"""Сравнение batch_size = 4, 8, 12 на РЕАЛЬНОЙ конфигурации ru60m
(18L/448d/vocab16k/head64, ctx_len=512), БЕЗ grad_checkpoint (реальный
рабочий режим -- скорость важнее памяти). compiled step идентичен
_make_step_simple из rwkv_metal/pretrain/trainer.py: mx.value_and_grad
-> clip_grad_norm -> AdamW.update, всё в одном mx.compile.

Синтетические случайные токены -- реальные train.txt/val.txt не нужны,
это чистый замер скорости/памяти, не качества.

Запуск (из корня rwkv-metal, тем же venv что и обучение):
    .venv/bin/python bench_batch_4_8_12.py

Останов по памяти: скрипт печатает peak после каждого B и сам
останавливается, если peak превышает MEM_STOP_GB (по умолчанию 11GB --
безопасный запас на 16GB машине с другими открытыми приложениями).
Если видите резкое замедление между запусками (не сразу после старта) --
это может быть своп; Ctrl+C безопасен в любой момент между батчами.
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.optimizers as optim
from rwkv_metal.pretrain.config import PretrainConfig
from rwkv_metal.model.rwkv7 import RWKV7, init_weights

GB = 1 / (1024 ** 3)
T = 512
VOCAB = 16000
BATCHES = [4, 8, 12]
MEM_STOP_GB = 11.0
ITERS = 8       # шагов на замер (после разогрева)
WARMUP = 3


def build(B):
    cfg = PretrainConfig(n_layer=18, n_embd=448, vocab_size=VOCAB,
                          head_size=64, ctx_len=T, batch_size=B,
                          grad_checkpoint=False)
    mx.random.seed(0)
    model = RWKV7(cfg)
    model = init_weights(model)
    model._grad_ckpt = False
    model.set_dtype("bfloat16")
    opt = optim.AdamW(learning_rate=1.5e-3)
    return model, opt


def make_step(model, opt):
    state = [model.state, opt.state]

    def _step(x, y):
        def loss_fn(m, x, y):
            return m.loss(x, y).astype(mx.float32)
        loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        opt.update(model, grads)
        return loss, norm

    return mx.compile(_step, inputs=state, outputs=state)


def run(B):
    model, opt = build(B)
    step = make_step(model, opt)

    mx.random.seed(1)
    x = mx.random.randint(0, VOCAB, (B, T))
    y = mx.random.randint(0, VOCAB, (B, T))
    mx.eval(x, y)

    for _ in range(WARMUP):
        loss, norm = step(x, y)
        mx.eval(loss, norm, model.state, opt.state)

    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        loss, norm = step(x, y)
        mx.eval(loss, norm, model.state, opt.state)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / ITERS

    peak = mx.get_peak_memory() * GB
    tok_s = B * T / dt

    del model, opt, step, x, y
    gc.collect()
    mx.clear_cache()

    return dt, tok_s, peak


def main():
    print("=== batch_size сравнение: 4 / 8 / 12 (grad_checkpoint=OFF, ru60m) ===")
    print(f"{'B':>4} {'step,ms':>10} {'tok/s':>10} {'peak,GB':>9} {'мкс/ток':>9} {'vs B=4':>8}")
    baseline_tok_s = None
    for B in BATCHES:
        try:
            dt, tok_s, peak = run(B)
        except Exception as e:
            print(f"B={B}: ОШИБКА {e}")
            break

        if baseline_tok_s is None:
            baseline_tok_s = tok_s
        scale = tok_s / baseline_tok_s

        per_tok_us = dt * 1e6 / (B * T)
        print(f"{B:>4} {dt*1e3:>10.1f} {tok_s:>10.0f} {peak:>9.2f} {per_tok_us:>9.3f} {scale:>7.2f}x")

        if peak > MEM_STOP_GB:
            print(f"\n-- остановка: peak {peak:.2f}GB > {MEM_STOP_GB}GB, дальше рискованно")
            break

    print("\nГотово. Если между двумя соседними B tok/s почти не растёт при "
          "заметном росте peak -- запас памяти на этот батч того не стоит; "
          "если tok/s растёт быстрее peak -- батч того стоит.")


if __name__ == "__main__":
    main()
