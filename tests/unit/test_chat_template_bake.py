"""Tests for baking a chat template into the finished GGUF.

Why this exists: in a GGUF the chat template lives INSIDE the container
(`tokenizer.chat_template`), so shipping a fixed `.jinja` beside the model does
nothing for llama.cpp users — and nothing for our own llama-server evals, which
read the template out of the file under test. These pin the wiring; none of them
need a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_tuner.config import RunConfig
from quant_tuner.quantize import gguf


def test_stamp_name_encodes_the_template_hash(tmp_path):
    """A changed template must produce a different stamp.

    `step()` is existence-based: a stamp named only after the stage would make
    swapping the template a silent no-op on re-run, benching the old template
    under the new label — the same trap the quant-GGUF naming convention avoids.
    """
    q = tmp_path / "IQ2_M-awq.gguf"
    a, b = tmp_path / "a.jinja", tmp_path / "b.jinja"
    a.write_text("{{ 'one' }}")
    b.write_text("{{ 'two' }}")

    assert gguf.template_stamp(q, a) != gguf.template_stamp(q, b)
    # ...and stable for the same bytes, or every re-run would re-patch 10 GB.
    b.write_text("{{ 'one' }}")
    assert gguf.template_stamp(q, a) == gguf.template_stamp(q, b)


def test_stamp_sits_beside_the_gguf_without_shadowing_it(tmp_path):
    q = tmp_path / "IQ2_M-awq.gguf"
    t = tmp_path / "t.jinja"
    t.write_text("x")
    stamp = gguf.template_stamp(q, t)
    assert stamp != q and stamp.name.startswith(q.name)


def test_stamp_distinguishes_general_name_too(tmp_path):
    """general.name rides in the same pass, so it must ride in the same stamp."""
    q = tmp_path / "m.gguf"
    t = tmp_path / "t.jinja"
    t.write_text("{{ 'x' }}")
    assert gguf.template_stamp(q, t, None) != gguf.template_stamp(q, t, "Qwen3.8-27B")
    assert gguf.template_stamp(q, t, "A") != gguf.template_stamp(q, t, "B")


def test_set_metadata_rejects_a_missing_template(tmp_path):
    """Fail before spawning the rewriter, not with a confusing subprocess error."""
    q = tmp_path / "m.gguf"
    q.write_bytes(b"GGUF")
    with pytest.raises(FileNotFoundError):
        gguf.set_metadata(q, template=tmp_path / "nope.jinja")
    assert not list(tmp_path.glob("*.tmp")), "must not leave a temp file behind"


def test_set_metadata_rejects_a_no_op_call(tmp_path):
    """Rewriting 10 GB to change nothing is a bug, not a default."""
    q = tmp_path / "m.gguf"
    q.write_bytes(b"GGUF")
    with pytest.raises(ValueError):
        gguf.set_metadata(q)


def test_set_metadata_leaves_the_original_intact_on_failure(tmp_path):
    """The GGUF is the run's expensive artifact — a failed patch must not eat it."""
    q = tmp_path / "m.gguf"
    q.write_bytes(b"not actually a gguf")
    t = tmp_path / "t.jinja"
    t.write_text("{{ 'x' }}")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        gguf.set_metadata(q, template=t)
    assert q.read_bytes() == b"not actually a gguf"
    assert not list(tmp_path.glob("*.tmp"))


def test_quantize_config_defaults_to_no_template_swap():
    """Baking a template is opt-in: it changes bytes users' servers render."""
    cfg = RunConfig.model_validate(
        {"name": "t", "model": "org/m", "workspace": "out/t",
         "data": {"logs": "l.jsonl"}}
    )
    assert cfg.quantize.chat_template is None
    assert cfg.quantize.general_name is None


def test_awq_recipe_bakes_the_verified_template():
    """The template file the recipe names must exist and be the A/B'd one."""
    recipe = (Path(__file__).parents[2] / "src/quant_tuner/recipes"
              / "iq2_m_qwen3_8_awq.yaml")
    cfg = RunConfig.from_yaml(recipe)
    assert cfg.quantize.chat_template is not None
    tmpl = Path(cfg.quantize.chat_template)
    assert tmpl.exists(), f"recipe names a template that is not in the repo: {tmpl}"


def test_pipeline_bakes_after_quantize_not_before():
    """Order matters: llama-quantize copies metadata from the F16 it reads.

    Patching before would be overwritten; patching after is what reaches the
    shipped file.
    """
    import inspect

    from quant_tuner import pipeline

    src = inspect.getsource(pipeline.quantize_model)
    assert src.index("llama-quantize") < src.index("chat_template"), (
        "the template bake must run after llama-quantize, or the quantizer's "
        "own metadata copy silently reverts it"
    )
