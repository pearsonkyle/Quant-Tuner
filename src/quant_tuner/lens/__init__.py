"""Jacobian-lens interpretability for GGUF quants.

Built on two upstreams (see the repo-root NOTICE):

- `anthropics/jacobian-lens <https://github.com/anthropics/jacobian-lens>`_
  (the ``vendor/jacobian-lens`` submodule) — the reference PyTorch
  implementation of the causal Jacobian lens; used optionally by
  ``quant-tuner lens fit-causal``.
- `igorbarshteyn/jlens-gguf <https://github.com/igorbarshteyn/jlens-gguf>`_
  (Apache-2.0, adapted in-tree) — the GGUF-native stack: the
  ``native/jlens_server`` activation/intervention server, the numpy lens
  math, and the D3 visualizer.

quant-tuner-specific layers on top:

- :mod:`quant_tuner.lens.backend`  jlens-server lifecycle (mirrors eval.server)
- :mod:`quant_tuner.lens.capture`  on-disk capture runs — the A/B unit
- :mod:`quant_tuner.lens.diff`     diff engine between two capture runs
- :mod:`quant_tuner.lens.replay`   tool-call decision-point replay
- :mod:`quant_tuner.lens.loops`    repetition detection + loop autopsy
- :mod:`quant_tuner.lens.probes`   knowledge probes (correct/suppressed/absent)
- :mod:`quant_tuner.lens.study`    "why calibration works" divergence study
- :mod:`quant_tuner.lens.gguf_edit` / :mod:`quant_tuner.lens.bake`
                                   weight-space ablation baked into a requant
- :mod:`quant_tuner.lens.cli`      the ``quant-tuner lens`` sub-app

Heavy imports (numpy, gguf-py) stay inside the submodules; importing this
package is cheap so the CLI can register lazily.
"""

__all__ = [
    "JacobianLensGGUF",
    "ReadoutWeights",
    "LensReadout",
    "NativeClient",
    "ForwardResult",
]


def __getattr__(name: str):  # lazy re-exports; keeps `import quant_tuner.lens` light
    if name in ("JacobianLensGGUF",):
        from quant_tuner.lens.lens import JacobianLensGGUF

        return JacobianLensGGUF
    if name in ("ReadoutWeights",):
        from quant_tuner.lens.model_reader import ReadoutWeights

        return ReadoutWeights
    if name in ("LensReadout",):
        from quant_tuner.lens.readout import LensReadout

        return LensReadout
    if name in ("NativeClient", "ForwardResult"):
        from quant_tuner.lens import client

        return getattr(client, name)
    raise AttributeError(name)
