from __future__ import annotations

import torch

from instant_lowpass import (
    InstantLowpassConfig,
    _AttachGraloraAdapterGradient,
    _AttachLowpassGraloraAdapterGradient,
    _InstantLowpassGraloraFunction,
    _InstantMergedGraloraFunction,
    _LowpassGraloraAdapterFunction,
    InstantLowpassLinear,
    _fallback_gralora_mix,
    _fallback_gralora_unmix,
    _gralora_delta_weight,
    _gralora_forward_from_x,
    _project_token_chunks,
    patch_gralora_with_instant_lowpass,
)


def _copy_linear(linear: torch.nn.Linear) -> torch.nn.Linear:
    copied = torch.nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    copied.load_state_dict(linear.state_dict())
    return copied


def _reference_gralora_forward(
    x: torch.Tensor,
    gralora_a: torch.Tensor,
    gralora_b: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, _in_features = x.shape
    block_count = int(gralora_a.shape[0])
    rank = int(gralora_a.shape[2])
    in_per_block = int(gralora_a.shape[1])
    sub_rank = rank // block_count
    hidden = torch.einsum(
        "blni,nir->blnr",
        x.view(batch, seq_len, block_count, in_per_block),
        gralora_a,
    )
    mixed = (
        hidden.view(batch, seq_len, block_count, block_count, sub_rank)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, seq_len, block_count, rank)
    )
    return torch.einsum("bljr,jro->bljo", mixed, gralora_b).reshape(batch, seq_len, -1)


def test_config_defaults_to_hadamard_32_of_64() -> None:
    config = InstantLowpassConfig()
    assert config.projector_kind == "hadamard"
    assert config.chunk_size == 64
    assert config.keep == 32
    assert config.parameter_gradient == "exact"


def test_config_normalizes_projected_lowpass_parameter_gradient() -> None:
    config = InstantLowpassConfig(parameter_gradient="projected-lowpass")
    assert config.parameter_gradient == "projected_lowpass"


def test_exact_linear_backward_matches_torch_linear() -> None:
    torch.manual_seed(5)
    exact = torch.nn.Linear(6, 4)
    wrapped = InstantLowpassLinear(
        _copy_linear(exact),
        InstantLowpassConfig(
            projector_kind="hadamard",
            chunk_size=4,
            keep=2,
        ),
    )
    x_exact = torch.randn(3, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    upstream = torch.randn(3, 8, 4)

    (exact(x_exact) * upstream).sum().backward()
    (wrapped(x_wrapped) * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(wrapped.weight.grad, exact.weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(wrapped.bias.grad, exact.bias.grad, rtol=1e-5, atol=1e-6)


def test_small_input_hidden_dim_uses_exact_linear_path() -> None:
    torch.manual_seed(3)
    exact = torch.nn.Linear(32, 128)
    wrapped = InstantLowpassLinear(
        _copy_linear(exact),
        InstantLowpassConfig(
            projector_kind="hadamard",
            chunk_size=4,
            keep=2,
            min_hidden_dim=64,
        ),
    )
    x = torch.randn(2, 8, 32, requires_grad=True)

    torch.testing.assert_close(wrapped(x), exact(x))
    assert int(wrapped.instant_calls.item()) == 0
    assert int(wrapped.fallback_calls.item()) == 1


def test_small_output_hidden_dim_uses_exact_linear_path() -> None:
    torch.manual_seed(4)
    exact = torch.nn.Linear(128, 32)
    wrapped = InstantLowpassLinear(
        _copy_linear(exact),
        InstantLowpassConfig(
            projector_kind="hadamard",
            chunk_size=4,
            keep=2,
            min_hidden_dim=64,
        ),
    )
    x = torch.randn(2, 8, 128, requires_grad=True)

    torch.testing.assert_close(wrapped(x), exact(x))
    assert int(wrapped.instant_calls.item()) == 0
    assert int(wrapped.fallback_calls.item()) == 1


def test_fallback_gralora_mix_matches_reference_layout() -> None:
    hidden = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    expected = torch.tensor(
        [
            [[0, 1, 4, 5], [2, 3, 6, 7]],
            [[8, 9, 12, 13], [10, 11, 14, 15]],
            [[16, 17, 20, 21], [18, 19, 22, 23]],
        ],
        dtype=torch.float32,
    )
    mixed = _fallback_gralora_mix(hidden)

    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(_fallback_gralora_unmix(mixed), hidden)


def test_gralora_forward_matches_reference() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 8, 6)
    gralora_a = torch.randn(2, 3, 4)
    gralora_b = torch.randn(2, 4, 5)

    expected = _reference_gralora_forward(x, gralora_a, gralora_b)
    actual = _gralora_forward_from_x(x, gralora_a, gralora_b, use_triton=False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gralora_delta_weight_matches_reference_forward() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 8, 6)
    gralora_a = torch.randn(2, 3, 4)
    gralora_b = torch.randn(2, 4, 5)

    expected = _reference_gralora_forward(x, gralora_a, gralora_b)
    actual = torch.nn.functional.linear(x, _gralora_delta_weight(gralora_a, gralora_b))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gralora_exact_adapter_grads_match_reference() -> None:
    torch.manual_seed(6)
    x_exact = torch.randn(2, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    gralora_a = torch.randn(2, 3, 4, requires_grad=True)
    gralora_b = torch.randn(2, 4, 5, requires_grad=True)
    gralora_a_wrapped = gralora_a.detach().clone().requires_grad_(True)
    gralora_b_wrapped = gralora_b.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 8, 10)

    exact = _reference_gralora_forward(x_exact, gralora_a, gralora_b)
    wrapped = _InstantLowpassGraloraFunction.apply(
        x_wrapped,
        gralora_a_wrapped,
        gralora_b_wrapped,
        False,
    )

    (exact * upstream).sum().backward()
    (wrapped * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_a_wrapped.grad, gralora_a.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_b_wrapped.grad, gralora_b.grad, rtol=1e-5, atol=1e-6)


def test_merged_gralora_exact_adapter_grads_match_reference() -> None:
    torch.manual_seed(8)
    scaling = 0.25
    x_exact = torch.randn(2, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    base_weight = torch.randn(10, 6)
    bias = torch.randn(10, requires_grad=True)
    bias_wrapped = bias.detach().clone().requires_grad_(True)
    gralora_a = torch.randn(2, 3, 4, requires_grad=True)
    gralora_b = torch.randn(2, 4, 5, requires_grad=True)
    gralora_a_wrapped = gralora_a.detach().clone().requires_grad_(True)
    gralora_b_wrapped = gralora_b.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 8, 10)

    exact = torch.nn.functional.linear(x_exact, base_weight, bias)
    exact = exact + scaling * _reference_gralora_forward(x_exact, gralora_a, gralora_b)
    merged_weight = (
        base_weight + scaling * _gralora_delta_weight(gralora_a_wrapped, gralora_b_wrapped)
    ).detach()
    wrapped = _InstantMergedGraloraFunction.apply(
        x_wrapped,
        merged_weight,
        bias_wrapped,
        gralora_a_wrapped,
        gralora_b_wrapped,
        scaling,
        False,
    )

    torch.testing.assert_close(wrapped, exact.detach(), rtol=1e-5, atol=1e-6)
    (exact * upstream).sum().backward()
    (wrapped * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias_wrapped.grad, bias.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_a_wrapped.grad, gralora_a.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_b_wrapped.grad, gralora_b.grad, rtol=1e-5, atol=1e-6)


def test_attach_gralora_adapter_gradient_keeps_native_linear_grad_x() -> None:
    torch.manual_seed(9)
    scaling = 0.5
    x_exact = torch.randn(2, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    base_weight = torch.randn(10, 6)
    bias = torch.randn(10, requires_grad=True)
    bias_wrapped = bias.detach().clone().requires_grad_(True)
    gralora_a = torch.randn(2, 3, 4, requires_grad=True)
    gralora_b = torch.randn(2, 4, 5, requires_grad=True)
    gralora_a_wrapped = gralora_a.detach().clone().requires_grad_(True)
    gralora_b_wrapped = gralora_b.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 8, 10)

    exact = torch.nn.functional.linear(x_exact, base_weight, bias)
    exact = exact + scaling * _reference_gralora_forward(x_exact, gralora_a, gralora_b)
    merged_weight = (
        base_weight + scaling * _gralora_delta_weight(gralora_a_wrapped, gralora_b_wrapped)
    ).detach()
    native = torch.nn.functional.linear(x_wrapped, merged_weight, bias_wrapped)
    wrapped = _AttachGraloraAdapterGradient.apply(
        native,
        x_wrapped,
        gralora_a_wrapped,
        gralora_b_wrapped,
        scaling,
        False,
    )

    torch.testing.assert_close(wrapped, exact.detach(), rtol=1e-5, atol=1e-6)
    (exact * upstream).sum().backward()
    (wrapped * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias_wrapped.grad, bias.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_a_wrapped.grad, gralora_a.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_b_wrapped.grad, gralora_b.grad, rtol=1e-5, atol=1e-6)


def test_project_token_chunks_stays_within_chunks() -> None:
    x = torch.arange(8, dtype=torch.float32).view(1, 8, 1)

    projected = _project_token_chunks(
        x,
        projector_kind="haar",
        chunk_size=4,
        keep=1,
        hadamard_backend="dense",
    )

    expected = torch.tensor([[[3.0]], [[11.0]]])
    torch.testing.assert_close(projected, expected)


def test_lowpass_attach_is_exact_when_keep_is_full_chunk() -> None:
    torch.manual_seed(10)
    scaling = 0.5
    x_exact = torch.randn(2, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    base_weight = torch.randn(10, 6)
    bias = torch.randn(10, requires_grad=True)
    bias_wrapped = bias.detach().clone().requires_grad_(True)
    gralora_a = torch.randn(2, 3, 4, requires_grad=True)
    gralora_b = torch.randn(2, 4, 5, requires_grad=True)
    gralora_a_wrapped = gralora_a.detach().clone().requires_grad_(True)
    gralora_b_wrapped = gralora_b.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 8, 10)

    exact = torch.nn.functional.linear(x_exact, base_weight, bias)
    exact = exact + scaling * _reference_gralora_forward(x_exact, gralora_a, gralora_b)
    merged_weight = (
        base_weight + scaling * _gralora_delta_weight(gralora_a_wrapped, gralora_b_wrapped)
    ).detach()
    native = torch.nn.functional.linear(x_wrapped, merged_weight, bias_wrapped)
    wrapped = _AttachLowpassGraloraAdapterGradient.apply(
        native,
        x_wrapped,
        gralora_a_wrapped,
        gralora_b_wrapped,
        scaling,
        False,
        "hadamard",
        8,
        8,
        "dense",
    )

    torch.testing.assert_close(wrapped, exact.detach(), rtol=1e-5, atol=1e-6)
    (exact * upstream).sum().backward()
    (wrapped * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias_wrapped.grad, bias.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_a_wrapped.grad, gralora_a.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gralora_b_wrapped.grad, gralora_b.grad, rtol=1e-5, atol=1e-5)


def test_unmerged_lowpass_gralora_is_exact_when_keep_is_full_chunk() -> None:
    torch.manual_seed(11)
    scaling = 0.5
    x_exact = torch.randn(2, 8, 6, requires_grad=True)
    x_wrapped = x_exact.detach().clone().requires_grad_(True)
    base = torch.nn.Linear(6, 10)
    base_wrapped = _copy_linear(base)
    for parameter in base_wrapped.parameters():
        parameter.requires_grad_(False)
    gralora_a = torch.randn(2, 3, 4, requires_grad=True)
    gralora_b = torch.randn(2, 4, 5, requires_grad=True)
    gralora_a_wrapped = gralora_a.detach().clone().requires_grad_(True)
    gralora_b_wrapped = gralora_b.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 8, 10)

    exact = base(x_exact) + scaling * _reference_gralora_forward(x_exact, gralora_a, gralora_b)
    adapter_output = _LowpassGraloraAdapterFunction.apply(
        x_wrapped,
        gralora_a_wrapped,
        gralora_b_wrapped,
        False,
        "hadamard",
        8,
        8,
        "dense",
    )
    wrapped = base_wrapped(x_wrapped) + scaling * adapter_output

    torch.testing.assert_close(wrapped, exact.detach(), rtol=1e-5, atol=1e-6)
    (exact * upstream).sum().backward()
    (wrapped * upstream).sum().backward()

    torch.testing.assert_close(x_wrapped.grad, x_exact.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gralora_a_wrapped.grad, gralora_a.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gralora_b_wrapped.grad, gralora_b.grad, rtol=1e-5, atol=1e-5)


def test_gralora_small_hidden_block_uses_original_forward() -> None:
    class FakeGralora(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = torch.nn.Linear(64, 128, bias=False)
            self.gralora_A = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.randn(2, 32, 128))}
            )
            self.gralora_B = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.randn(2, 128, 64))}
            )
            self.gralora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
            self.gralora_k = 2
            self.hybrid_r = {"default": 0}
            self.active_adapters = ["default"]
            self.scaling = {"default": 1.0}
            self.disable_adapters = False
            self.merged = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_layer(x)

    model = torch.nn.Sequential(FakeGralora())
    patched = patch_gralora_with_instant_lowpass(
        model,
        InstantLowpassConfig(
            projector_kind="hadamard",
            chunk_size=64,
            keep=16,
            min_hidden_dim=64,
        ),
    )
    x = torch.randn(1, 64, 64, requires_grad=True)

    assert patched == ["0"]
    torch.testing.assert_close(model(x), model[0].base_layer(x))
    assert int(model[0].instant_lowpass_instant_calls.item()) == 0
    assert int(model[0].instant_lowpass_fallback_calls.item()) == 1
