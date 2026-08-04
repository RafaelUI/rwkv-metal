import copy

try:
    from rwkv_quant.api import PRESETS, quantize
    from rwkv_quant.calibration.group_config import QuantConfig
except ModuleNotFoundError as exc:
    raise SystemExit(
        "rwkv_quant is not installed in this Python environment. "
        "Install it with `python3 -m pip install rwkv-quant` or run this script from an environment where the rwkv-quant package is available."
    ) from exc

if __name__ == "__main__":
    # Use reduction-style quantization, but do not require the external
    # activation stats file that rwkv_quant preset defaults expect.
    config = copy.deepcopy(PRESETS["compression"])
    config.act_stats_path = None

    quantize(
        "/Users/s/Develop/rwkv-metal/model.pth",
        "model.rwkvq",
        config=config,
    )
