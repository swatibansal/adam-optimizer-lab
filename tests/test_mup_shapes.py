"""Shape / grouping / scaling checks for the tiny model and µP helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import set_seed
from optlab.mup import apply_mup_init, build_param_groups, lr_multipliers
from optlab.tiny_model import TinyLM, group_of

WIDTHS = [256, 512, 1024]
BASE_WIDTH = 256


def test_instantiates_at_every_width() -> None:
    for width in WIDTHS:
        set_seed(0)
        model = TinyLM(width)
        x = torch.randint(0, model.vocab, (2, model.seq_len))
        logits = model(x)
        assert logits.shape == (2, model.seq_len, model.vocab)


def test_every_parameter_is_grouped() -> None:
    model = TinyLM(256)
    expected = {"embed", "block0.attn", "block0.mlp", "block1.attn", "block1.mlp", "head"}
    seen = {group_of(name) for name, _ in model.named_parameters()}
    assert seen == expected
    # grouped_parameters covers every parameter exactly once.
    grouped = model.grouped_parameters()
    total = sum(len(v) for v in grouped.values())
    assert total == len(list(model.parameters()))


def test_mup_lr_multipliers() -> None:
    for width in WIDTHS:
        model = TinyLM(width)
        mult = lr_multipliers(model, BASE_WIDTH, mup=True)
        # Embedding and head keep the base LR.
        assert mult["embed"] == 1.0
        assert mult["head"] == 1.0
        # Hidden layers scale as base_width / width.
        assert mult["block0.attn"] == BASE_WIDTH / width
        assert mult["block0.mlp"] == BASE_WIDTH / width
        # Standard parameterization leaves everything at 1.0.
        assert all(v == 1.0 for v in lr_multipliers(model, BASE_WIDTH, mup=False).values())


def test_mup_param_group_learning_rates() -> None:
    width = 1024
    base_lr = 1e-2
    model = TinyLM(width)
    groups = build_param_groups(model, base_lr, BASE_WIDTH, weight_decay=0.01, mup=True)
    by_name = {g["name"]: g for g in groups}
    assert by_name["embed"]["lr"] == base_lr
    assert by_name["head"]["lr"] == base_lr
    assert by_name["block0.attn"]["lr"] == base_lr * BASE_WIDTH / width


def test_mup_model_scales_and_forward() -> None:
    base = 256
    for width in WIDTHS:
        set_seed(0)
        m = TinyLM(width, mup=True, base_width=base)
        # Attention scale is width-invariant (sqrt(base)/width); output multiplier base/width.
        assert m.attn_scale == pytest.approx(base**0.5 / width)
        assert m.out_mult == pytest.approx(base / width)
        x = torch.randint(0, m.vocab, (2, m.seq_len))
        assert m(x).shape == (2, m.seq_len, m.vocab)
    # At base width the µP attention scale coincides with the standard 1/sqrt(width).
    assert TinyLM(base, mup=True, base_width=base).attn_scale == pytest.approx(1.0 / base**0.5)
    # The standard model keeps 1/sqrt(width) and no output multiplier.
    std = TinyLM(512)
    assert std.attn_scale == pytest.approx(1.0 / 512**0.5)
    assert std.out_mult == 1.0


def test_mup_init_shrinks_hidden_variance() -> None:
    set_seed(0)
    wide = TinyLM(1024)
    hidden_std_before = wide.blocks[0].mlp.fc1.weight.std().item()
    apply_mup_init(wide, BASE_WIDTH)
    hidden_std_after = wide.blocks[0].mlp.fc1.weight.std().item()
    # width 1024 vs base 256 -> std scaled by sqrt(256/1024) = 0.5.
    assert hidden_std_after < hidden_std_before
    assert hidden_std_after == pytest.approx(0.02 * (BASE_WIDTH / 1024) ** 0.5, rel=0.15)
