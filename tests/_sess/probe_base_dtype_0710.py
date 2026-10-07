"""07.10: тип квантованной базы в rwkv-metal -- bf16 (как было) против fp16 (норма rwkv-quant), и каст входа в слое.
Плечи переключаются флагами rl.BASE_DTYPE / rl.NOCAST на ОДНОЙ модели в одном процессе (закон 27):
  bf16        база bf16, матмул слоя в bf16 (как было);
  fp16        база fp16, матмул слоя в fp16;
  bf16_nocast база bf16, вход не приводится (fp32-вход тянет fp32-матмул) -- прежнее поведение до 22.08;
  fp16_nocast база fp16, вход не приводится -- самое точное, эталон для градиентов.
Меряется: (1) KL(истина || плечо) прямого прохода, истина -- исходный чекпоинт, веса bf16, счёт fp32 (RWKV7Ref, cpu);
(2) градиенты адаптеров (lora_b возмущены, чтобы градиент по lora_a не был нулём; потери -- средняя CE по T токенам,
как в обучении): отн. L2-ошибка и косинус к эталону fp16_nocast, доля нулевых и неконечных элементов;
(3) с --speed: шаг обучения A/B bf16 против fp16 (чередование, случайный порядок, своп на границе).
    python probe_base_dtype_0710.py <файл.rwkvq> <ckpt.pth> <out.json> [окон=8] [T=256] [--speed [T раундов]]"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from rwkv_metal.lora import load_rwkvq_model, rwkvq_linear as rl
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FILE, CK, OUT = args[:3]; NW = int(args[3]) if len(args) > 3 else 8; T = int(args[4]) if len(args) > 4 else 256
SPEED = "--speed" in sys.argv; ST = int(args[5]) if len(args) > 5 else 512; ROUNDS = int(args[6]) if len(args) > 6 else 4
def swap_mb():
    p = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))
ARMS = {"bf16": (mx.bfloat16, False, None), "fp16": (mx.float16, False, None), "fp16_bbf16": (mx.float16, False, mx.bfloat16),
        "fp16_bfp32": (mx.float16, False, mx.float32), "bf16_nocast": (mx.bfloat16, True, None), "fp16_nocast": (mx.float16, True, None)}
def use(a): rl.BASE_DTYPE, rl.NOCAST, rl.BWD_DTYPE = ARMS[a]
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"]
rows = [int(i) for i in np.linspace(0, ev.shape[0] - 1, NW).round()]
WINS = [ev[i, :T] for i in rows]
RES = {"file": FILE, "kl": {}, "grad": {}, "speed": {}}
sw0 = swap_mb()
if not SPEED:
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    def lsm(a): return torch.log_softmax(torch.as_tensor(np.asarray(a, dtype=np.float32)), -1)
    def kl(p, q): return float((p.exp() * (p - q)).sum(-1).mean())
    MO = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
    with torch.no_grad(): TRUTH = [torch.log_softmax(MO.forward(w[None], cfg=None).float(), -1)[0] for w in WINS]
    del MO
model, cfg, info = load_rwkvq_model(FILE, rank=16, verbose=False)
print("базы:", sorted({type(m).__name__ for _, m in model.named_modules() if type(m).__name__.startswith("Rwkvq")}), flush=True)
if not SPEED:
    def logits(w):
        lg = model(mx.array(w.numpy().astype(np.int32)[None])); lg = (lg[0] if lg.ndim == 3 else lg).astype(mx.float32); mx.eval(lg); return np.array(lg)
    for a in ARMS:
        use(a); RES["kl"][a] = [kl(TRUTH[i], lsm(logits(w))) for i, w in enumerate(WINS)]
        print("KL %-12s %.6f" % (a, np.mean(RES["kl"][a])), flush=True)
    A = {k: np.array(v) for k, v in RES["kl"].items()}; rng = np.random.default_rng(0); idx = rng.integers(0, NW, size=(20000, NW))
    def ratio(a, b):
        r = A[a][idx].mean(1) / A[b][idx].mean(1) - 1
        return "%+.2f%% [%+.2f; %+.2f]" % (100 * (A[a].mean() / A[b].mean() - 1), 100 * np.quantile(r, .025), 100 * np.quantile(r, .975))
    for a, b in (("bf16", "fp16_nocast"), ("fp16", "fp16_nocast"), ("bf16_nocast", "fp16_nocast"), ("fp16", "bf16")):
        print("   KL %-12s / %-12s %s" % (a, b, ratio(a, b)), flush=True)
    # ---- градиенты ----
    rs = np.random.RandomState(3)
    params = dict(tree_flatten(model.trainable_parameters()))
    upd = [(k, (v.astype(mx.float32) + mx.array(rs.normal(0, 1e-3, v.shape).astype(np.float32))).astype(v.dtype)) for k, v in params.items() if k.endswith("lora_b")]
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(upd)); mx.eval(model.parameters())
    w = WINS[0]; x = mx.array(w[:-1].numpy().astype(np.int32)[None]); y = mx.array(w[1:].numpy().astype(np.int32)[None])
    gf = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))
    G = {}
    for a in ARMS:
        use(a); loss, g = gf(model, x, y); flat = dict(tree_flatten(g)); mx.eval(loss, flat)
        G[a] = (float(loss), {k: np.array(v.astype(mx.float32)).ravel() for k, v in flat.items()})
    ref = np.concatenate([G["fp16_nocast"][1][k] for k in sorted(G["fp16_nocast"][1])])
    for a in ARMS:
        v = np.concatenate([G[a][1][k] for k in sorted(G[a][1])])
        r = dict(loss=G[a][0], rel_l2=float(np.linalg.norm(v - ref) / np.linalg.norm(ref)), cos=float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-30)),
                 zeros=float((v == 0).mean()), nonfinite=int((~np.isfinite(v)).sum()), norm=float(np.linalg.norm(v)), n=int(v.size))
        RES["grad"][a] = r
        print("ГРАД %-12s потери %.5f | отн. L2 к эталону %.3e, косинус %.6f | нулей %.4f%%, неконечных %d | норма %.4e (эл-тов %d)"
              % (a, r["loss"], r["rel_l2"], r["cos"], 100 * r["zeros"], r["nonfinite"], r["norm"], r["n"]), flush=True)
else:
    model._grad_ckpt = True
    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, ST)).astype(np.int32)); y = mx.array(rs.randint(1, 60000, size=(1, ST)).astype(np.int32))
    gf = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32)); opt = optim.AdamW(learning_rate=1e-4)
    def step():
        loss, grads = gf(model, x, y); opt.update(model, grads); mx.eval(loss, model.state, opt.state); return float(loss)
    VAR = ["bf16", "fp16", "fp16_bbf16", "fp16_bfp32"]; acc = {n: [] for n in VAR}; pk = {n: 0.0 for n in VAR}; rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = [VAR[i] for i in rng.permutation(len(VAR))]
        for a in order:
            use(a); step(); mx.clear_cache(); mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter(); l = step(); acc[a].append(time.perf_counter() - t1)
            assert np.isfinite(l), (a, l)
            pk[a] = max(pk[a], mx.get_peak_memory() / 1e6)
    for a in VAR:
        m = float(np.median(acc[a])); RES["speed"][a] = dict(ms=m * 1e3, spread=(max(acc[a]) - min(acc[a])) / m * 100, peak_mb=pk[a], all=acc[a])
        print("ШАГ %-5s %7.0f мс (разброс %.1f%%), пик %.0f МБ" % (a, m * 1e3, RES["speed"][a]["spread"], pk[a]), flush=True)
    for a in VAR[1:]:
        print("%s против bf16: %+.1f%% времени, пик %+.0f МБ (T=%d, раундов %d)" % (a, 100 * (RES["speed"][a]["ms"] / RES["speed"]["bf16"]["ms"] - 1), pk[a] - pk["bf16"], ST, ROUNDS))
sw1 = swap_mb(); RES["swap"] = [sw0, sw1]
print("своп за прогон %+.0f МБ" % (sw1 - sw0))
json.dump(RES, open(OUT, "w"), indent=1)
