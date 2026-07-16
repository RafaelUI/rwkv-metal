"""Regression tests for the WKV-7 Metal backward kernels."""

import numpy as np

import mlx.core as mx

from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint


def test_checkpoint_backward_is_deterministic_across_eager_and_compiled():
    """Repeated backward dispatches must produce bitwise-identical gradients.

    The kernel reuses its threadgroup arrays at every reverse timestep. Without
    a barrier after the final C_row update, one SIMD group can overwrite the
    next timestep's shared values while another is still reading the current
    w/a vectors. Compiling the graph changes scheduling enough to expose the
    race reliably on Apple Silicon.
    """
    batch, time, heads, dim = 1, 512, 24, 64
    mx.random.seed(77)
    inputs = (
        (mx.random.normal((batch, time, heads, dim)) * 0.3).astype(mx.bfloat16),
        (
            mx.sigmoid(mx.random.normal((batch, time, heads, dim)) * 1.2)
            * 0.25
            + 0.74
        ).astype(mx.bfloat16),
        (mx.random.normal((batch, time, heads, dim)) * 0.3).astype(mx.bfloat16),
        (mx.random.normal((batch, time, heads, dim)) * 0.3).astype(mx.bfloat16),
        (mx.random.normal((batch, time, heads, dim)) * 0.09).astype(mx.bfloat16),
        (mx.random.normal((batch, time, heads, dim)) * 0.09).astype(mx.bfloat16),
    )
    mx.eval(*inputs)
    kernel = make_wkv7_checkpoint(batch, time, heads, dim)

    def loss_fn(*args):
        output = kernel(*args)
        weights = mx.linspace(0.5, 1.5, time)[None, :, None, None]
        return mx.mean(output * weights)

    eager = mx.value_and_grad(loss_fn, argnums=list(range(6)))
    compiled = mx.compile(mx.value_and_grad(loss_fn, argnums=list(range(6))))

    def run(grad_fn):
        loss, gradients = grad_fn(*inputs)
        mx.eval(loss, *gradients)
        return float(loss), tuple(
            np.array(gradient.astype(mx.float32), copy=True)
            for gradient in gradients
        )

    runs = [run(eager), run(eager), run(compiled), run(compiled), run(compiled)]
    reference_loss, reference_gradients = runs[0]
    for loss, gradients in runs[1:]:
        assert loss == reference_loss
        for name, reference, actual in zip(
            ("r", "w", "k", "v", "a", "b"), reference_gradients, gradients
        ):
            np.testing.assert_array_equal(
                actual,
                reference,
                err_msg=f"gradient {name} changed across backward dispatches",
            )
