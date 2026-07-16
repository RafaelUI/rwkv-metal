"""Golden correctness and determinism tests for WKV-7 Metal backward."""

import numpy as np

import mlx.core as mx

from rwkv_metal.kernel import (
    make_wkv7_checkpoint_with_state,
    wkv7_train_py_with_state,
)


B, T, H, D = 2, 64, 4, 64
GRADIENT_NAMES = ("dr", "dw", "dk", "dv", "da", "db", "dh_in")
GOLDEN_REL_ERR = 1e-5
RACE_B, RACE_T, RACE_H, RACE_D = 1, 512, 24, 64


def _make_inputs():
    """Create fp32 inputs with RWKV-7-like decay and a nonzero initial state."""
    mx.random.seed(0)
    shape = (B, T, H, D)
    r, k, v = tuple(mx.random.normal(shape) * 0.5 for _ in range(3))
    # The model passes a=-kk into WKV, where kk is L2-normalized per head.
    a = mx.random.normal(shape)
    a = a / mx.sqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-12)
    b = -a * mx.random.uniform(0.9, 1.1, shape=shape)
    # RWKV-7 uses a double-exponential decay whose learned logits have a strong
    # negative bias. This produces a realistic non-uniform range (~0.58-0.99)
    # without near-zero values that make inverse checkpoint reconstruction
    # intentionally ill-conditioned.
    w = mx.exp(-mx.exp(mx.random.normal(shape) * 0.5 - 2.5))
    h_in = mx.random.normal((B, H, D, D)) * 0.1
    p_out = mx.random.normal(shape)
    p_h = mx.random.normal((B, H, D, D))
    mx.eval(r, w, k, v, a, b, h_in, p_out, p_h)
    return (r, w, k, v, a, b, h_in), p_out, p_h


def _reference_with_state(r, w, k, v, a, b, h_in):
    """CPU einsum reference with explicit nonzero h_in and returned h_out.

    MLX's GPU einsum reduction is optimized for throughput and differs from a
    scalar fp32 accumulation by roughly 1e-3 for this workload. Running the
    reference on the CPU keeps its reduction error below the 1e-5 golden line,
    so the metric measures the custom Metal kernel rather than the reference.
    """
    with mx.stream(mx.cpu):
        return wkv7_train_py_with_state(r, w, k, v, a, b, h_in)


def _loss_fn(forward, p_out, p_h):
    def loss(r, w, k, v, a, b, h_in):
        out, h_out = forward(r, w, k, v, a, b, h_in)
        return (out * p_out).sum() + (h_out * p_h).sum()

    return loss


def _gradient_functions():
    inputs, p_out, p_h = _make_inputs()
    metal = make_wkv7_checkpoint_with_state(B, T, H, D)
    metal_grad = mx.compile(
        mx.grad(_loss_fn(metal, p_out, p_h), argnums=list(range(7)))
    )
    reference_grad = mx.grad(
        _loss_fn(_reference_with_state, p_out, p_h), argnums=list(range(7))
    )
    return inputs, metal_grad, reference_grad


def test_backward_gradients_match_einsum_reference():
    """The worst normalized max error across all seven gradients stays <1e-5."""
    inputs, metal_grad, reference_grad = _gradient_functions()
    metal_gradients = metal_grad(*inputs)
    reference_gradients = reference_grad(*inputs)
    mx.eval(*metal_gradients, *reference_gradients)

    relative_errors = {}
    for name, actual, expected in zip(
        GRADIENT_NAMES, metal_gradients, reference_gradients
    ):
        scale = float(mx.abs(expected).max()) + 1e-30
        max_error = float(mx.abs(actual - expected).max())
        relative_errors[name] = max_error / scale

    worst_name = max(relative_errors, key=relative_errors.get)
    worst = relative_errors[worst_name]
    details = ", ".join(
        f"{name}={relative_errors[name]:.2e}" for name in GRADIENT_NAMES
    )
    assert worst < GOLDEN_REL_ERR, (
        f"GOLDEN METRIC FAILED: {worst_name}={worst:.2e}; {details}"
    )


def test_backward_is_bitwise_deterministic():
    """Large independent dispatches return bitwise-identical gradients.

    The T=64/H=4 golden fixture is deliberately small for the CPU reference,
    but is not stressful enough to expose the missing-barrier race reliably.
    This regression instead uses the original T=512/H=24 bf16 workload, runs
    two eager plus three compiled backwards, and recreates every primal before
    each evaluated call so every sample requires an independent Metal dispatch.
    """
    mx.random.seed(77)
    shape = (RACE_B, RACE_T, RACE_H, RACE_D)
    race_inputs = (
        (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
        (
            mx.sigmoid(mx.random.normal(shape) * 1.2) * 0.25 + 0.74
        ).astype(mx.bfloat16),
        (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
        (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
        (mx.random.normal(shape) * 0.09).astype(mx.bfloat16),
        (mx.random.normal(shape) * 0.09).astype(mx.bfloat16),
    )
    h_in = (mx.random.normal((RACE_B, RACE_H, RACE_D, RACE_D)) * 0.05).astype(
        mx.float32
    )
    mx.eval(*race_inputs, h_in)

    # Store canonical fp32 host copies. Each run reconstructs fresh bf16 MLX
    # arrays from these values, so lazy graph reuse cannot satisfy two samples
    # with a single previously materialized backward dispatch.
    input_values = tuple(
        np.array(value.astype(mx.float32), copy=True) for value in race_inputs
    )
    h_in_value = np.array(h_in, copy=True)
    kernel = make_wkv7_checkpoint_with_state(
        RACE_B, RACE_T, RACE_H, RACE_D
    )

    def loss_fn(r, w, k, v, a, b, state):
        output, h_out = kernel(r, w, k, v, a, b, state)
        time_weights = mx.linspace(0.5, 1.5, RACE_T)[None, :, None, None]
        state_weights = mx.linspace(-0.3, 0.7, RACE_D)[None, None, None, :]
        return mx.mean(output * time_weights) + mx.mean(h_out * state_weights)

    eager = mx.grad(loss_fn, argnums=list(range(7)))
    compiled = mx.compile(mx.grad(loss_fn, argnums=list(range(7))))

    def run(grad_fn):
        current_inputs = tuple(
            mx.array(value).astype(mx.bfloat16) for value in input_values
        )
        current_h_in = mx.array(h_in_value)
        gradients = grad_fn(*current_inputs, current_h_in)
        mx.eval(*gradients)
        return tuple(
            np.array(gradient.astype(mx.float32), copy=True)
            for gradient in gradients
        )

    runs = [run(eager), run(eager), run(compiled), run(compiled), run(compiled)]
    reference = runs[0]
    for run_index, gradients in enumerate(runs[1:], start=2):
        for name, expected, actual in zip(
            GRADIENT_NAMES, reference, gradients
        ):
            np.testing.assert_array_equal(
                actual,
                expected,
                err_msg=(
                    f"gradient {name} changed on independent backward "
                    f"dispatch {run_index}"
                ),
            )
