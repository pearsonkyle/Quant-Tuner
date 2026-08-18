"""``wrap_model``: which linears go on the ternary grid, and which get gradients.

On a natively-ternary model (Ternary-Bonsai) those two questions have the same answer —
every weight already sits on the grid, so a frozen layer is ternary for free and the only
choice is what to train. gemma-4 is dense, which splits them: a progressive schedule needs
layers that are **still bf16 and not being moved to the grid yet**, and without that third
state every layer is ternarized at step 0, which is the all-at-once approach the schedule
exists to avoid.

These use a synthetic nested module tree so they need no checkpoint. The nesting is not
incidental: ``Gemma4ForConditionalGeneration`` keeps its decoder at
``model.language_model.layers``, and the old hard-coded ``model.model.layers`` raised on it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from quant_tuner.qat.ternary import TernaryLinear
from quant_tuner.qat.train import decoder_layers, wrap_model


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(256, 256, bias=False)
        self.mlp.down_proj = nn.Linear(256, 256, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(256, 256, bias=False)


def _model(nested: bool, n: int = 6) -> nn.Module:
    """``nested`` mirrors gemma-4's model.language_model.layers; else a plain CausalLM."""
    m = nn.Module()
    m.model = nn.Module()
    layers = nn.ModuleList([_Block() for _ in range(n)])
    if nested:
        m.model.language_model = nn.Module()
        m.model.language_model.layers = layers
    else:
        m.model.layers = layers
    return m


@pytest.mark.parametrize("nested", [True, False])
def test_decoder_layers_finds_both_layouts(nested: bool) -> None:
    assert len(decoder_layers(_model(nested))) == 6


def test_decoder_layers_refuses_an_unknown_layout() -> None:
    """Better than returning None and training nothing, silently."""
    with pytest.raises(AttributeError, match="no decoder layer list"):
        decoder_layers(nn.Linear(4, 4))


def test_all_layers_ternarized_by_default() -> None:
    """The Bonsai behaviour must be unchanged when no schedule is given."""
    m = _model(nested=False)
    wrap_model(m, n_train=2)
    kinds = [type(b.mlp.gate_proj).__name__ for b in decoder_layers(m)]
    assert kinds == ["TernaryLinear"] * 6


def test_ternary_layers_leaves_the_rest_dense() -> None:
    m = _model(nested=True)
    wrap_model(m, n_train=2, ternary_spec="4-5")
    layers = decoder_layers(m)
    for i, b in enumerate(layers):
        expected = TernaryLinear if i >= 4 else nn.Linear
        assert isinstance(b.mlp.gate_proj, expected), i
    # a dense, untrained layer must not accumulate gradients either
    assert not layers[0].mlp.gate_proj.weight.requires_grad


def test_dense_kind_is_excluded_in_every_ternarized_layer() -> None:
    """The measured gemma-4 case: keep mlp.down_proj off the grid, ternarize the rest."""
    m = _model(nested=True)
    wrap_model(m, n_train=6, dense_kinds=("down_proj",))
    for b in decoder_layers(m):
        assert isinstance(b.mlp.down_proj, nn.Linear)
        assert not isinstance(b.mlp.down_proj, TernaryLinear)
        assert isinstance(b.mlp.gate_proj, TernaryLinear)
        assert isinstance(b.self_attn.q_proj, TernaryLinear)


def test_a_dense_kind_inside_a_trainable_layer_still_trains() -> None:
    """This is most of why a partial schedule beats all-at-once: the tensors left dense
    have to be free to adapt to their ternarized neighbours. They are plain
    ``Linear.weight``s, not ``.linear.weight`` latents, so the name-based requires_grad
    pass cannot see them — a regression here silently freezes them."""
    m = _model(nested=True)
    wrap_model(m, n_train=2, dense_kinds=("down_proj",))
    layers = decoder_layers(m)
    assert layers[5].mlp.down_proj.weight.requires_grad      # trainable layer
    assert not layers[0].mlp.down_proj.weight.requires_grad  # frozen layer


def test_training_a_layer_that_is_not_ternarized_is_refused() -> None:
    """Gradients on a latent nothing quantizes is plain fine-tuning wearing a QAT label."""
    with pytest.raises(ValueError, match="train but not to ternarize"):
        wrap_model(_model(nested=True), n_train=6, ternary_spec="4-5")


def test_dense_layers_keep_their_exact_weights() -> None:
    """A layer outside the schedule must be untouched — not ternarized, not rescaled."""
    m = _model(nested=True)
    before = decoder_layers(m)[0].mlp.gate_proj.weight.detach().clone()
    wrap_model(m, n_train=2, ternary_spec="4-5")
    assert torch.equal(decoder_layers(m)[0].mlp.gate_proj.weight, before)


class _TowerBlock(torch.nn.Module):
    """Mimics gemma-4's vision/audio encoder blocks: their own ``layers.N`` index AND
    submodules literally named ``linear``, which is what made the old name-based
    requires_grad rule claim them."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Module()
        self.self_attn.q_proj.linear = nn.Linear(256, 256, bias=False)


def _multimodal(n: int = 6) -> nn.Module:
    m = _model(nested=True, n=n)
    m.model.vision_tower = nn.Module()
    m.model.vision_tower.encoder = nn.Module()
    m.model.vision_tower.encoder.layers = nn.ModuleList([_TowerBlock() for _ in range(n)])
    return m


def test_towers_are_never_marked_trainable() -> None:
    """The bug this pins cost 167.8 M params of gemma-4's vision+audio towers being
    handed to the optimizer. Nothing failed loudly — a text-only forward never gives
    them a gradient — so they simply consumed optimizer state and took weight decay."""
    m = _multimodal()
    wrap_model(m, n_train=2)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "sanity: something should train"
    assert not [n for n in trainable if "vision_tower" in n], trainable


def test_towers_are_not_ternarized_either() -> None:
    m = _multimodal()
    wrap_model(m, n_train=6)
    tower = m.model.vision_tower.encoder.layers[0].self_attn.q_proj
    assert isinstance(tower.linear, nn.Linear)
    assert not isinstance(tower.linear, TernaryLinear)
