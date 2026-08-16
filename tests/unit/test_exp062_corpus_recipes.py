"""Wiring tests for the exp-062 rebuild (new corpus + AWQ rungs).

Two things are pinned here, both of which fail silently rather than loudly:

1. The corpus builder's DEFAULTS still reproduce the pinned exp-060 corpus. One
   script now builds both that corpus and the re-cut exp-062 one; if a default
   drifts, every published number attributed to the original silently stops being
   reproducible, with nothing to notice it.

2. The exp-062 recipes point at the NEW corpus and the verified template. `step()`
   is existence-based, so a recipe left pointing at the old corpus path would
   happily reuse stale artifacts and bench them under the new label.

None of these need a model or a GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quant_tuner.config import RunConfig

REPO = Path(__file__).resolve().parents[2]
RUNGS = {
    "iq2_m_qwen3_8_awq_v2": ("IQ2_M", "out/exp-062-awq-iq2m"),
    "iq3_m_qwen3_8_awq": ("IQ3_M", "out/exp-062-awq-iq3m"),
    "iq4_xs_qwen3_8_awq": ("IQ4_XS", "out/exp-062-awq-iq4xs"),
}
NEW_CORPUS = "out/exp-062-32k/corpora/corpus.cal.txt"
TEMPLATE = "data/chat_templates/qwen3_8_safe_v2.jinja"


def _builder():
    """Load the repack script by path — scripts/ is not an importable package."""
    path = REPO / "scripts" / "exp060_repack_cal_32k.py"
    spec = importlib.util.spec_from_file_location("exp060_repack_cal_32k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_flags_reproduces_the_pinned_exp060_budgets():
    """The whole reproducibility claim for the published corpus rests on this."""
    mod = _builder()
    budgets, dropped = mod.resolve_budgets([], [])
    assert dropped == set(), "a default drop would silently re-cut the pinned corpus"
    assert budgets == {
        "logs": 2_000_000,
        "reasoning": 1_000_000,
        "swe-trajectories": 1_000_000,
        "broad-supplement": None,
        "redteam-refusals": None,
        "wiki": None,
    }


def test_budget_and_drop_overrides_apply():
    mod = _builder()
    budgets, dropped = mod.resolve_budgets(
        ["logs=2800000", "broad-supplement=300000"],
        ["redteam-refusals", "reasoning"],
    )
    assert budgets["logs"] == 2_800_000
    assert budgets["broad-supplement"] == 300_000
    assert dropped == {"redteam-refusals", "reasoning"}
    # untouched strata keep their defaults
    assert budgets["swe-trajectories"] == 1_000_000


def test_budget_none_means_uncapped():
    budgets, _ = _builder().resolve_budgets(["logs=none"], [])
    assert budgets["logs"] is None


@pytest.mark.parametrize("spec", ["logs", "nosuchstratum=5", "logs=notanint"])
def test_malformed_budget_specs_raise(spec):
    """A typo must not be a silent no-op — it would contradict the written audit."""
    with pytest.raises(ValueError):
        _builder().resolve_budgets([spec], [])


def test_unknown_drop_raises():
    with pytest.raises(ValueError):
        _builder().resolve_budgets([], ["redteam_refusals"])  # underscore typo


@pytest.mark.parametrize("recipe,expected", sorted(RUNGS.items()))
def test_exp062_recipes_use_the_new_corpus_and_template(recipe, expected):
    quant_type, workspace = expected
    cfg = RunConfig.from_yaml(REPO / "src/quant_tuner/recipes" / f"{recipe}.yaml")
    assert cfg.quantize.type == quant_type
    assert str(cfg.workspace).endswith(workspace.split("/")[-1])
    # The corpus swap is the point of the rebuild; a stale path would reuse
    # exp-060 artifacts under an exp-062 label.
    assert str(cfg.data.corpus).endswith(NEW_CORPUS.split("/", 1)[1])
    assert str(cfg.quantize.chat_template).endswith(TEMPLATE.split("/")[-1])
    assert cfg.quantize.general_name == "Qwen3.8-27B"
    assert cfg.quantize.mtp_pin == "q8_0"


@pytest.mark.parametrize("recipe", sorted(RUNGS))
def test_exp062_recipes_pin_ctx_to_the_corpus_packing(recipe):
    """ctx is a PACKING parameter: the corpus was cut for 32768 and only 32768.

    Reading it at another ctx does not calibrate on 32K trajectories — it glues
    unrelated windows into one context, with no error to notice.
    """
    cfg = RunConfig.from_yaml(REPO / "src/quant_tuner/recipes" / f"{recipe}.yaml")
    params = cfg.calibration.params
    assert params["ctx"] == 32768
    assert params["imatrix_ctx"] == 32768
    # Eval stays at 8192 so the 143 GB of FP16 baselines remain reusable.
    assert cfg.bench.eval_ctx == 8192
