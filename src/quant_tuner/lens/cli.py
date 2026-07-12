"""``quant-tuner lens`` — Jacobian-lens interpretability for GGUF quants.

Heavy imports live inside command bodies (repo convention), so the sub-app
registers without pulling numpy/gguf on every CLI invocation.
"""

from __future__ import annotations

from pathlib import Path

import typer

lens_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Jacobian-lens interpretability for GGUF quants (inspect, A/B, diagnose, bake).",
)


@lens_app.command("build-server")
def build_server(
    llama_dir: Path | None = typer.Option(None, help="llama.cpp checkout (default: vendor/llama.cpp)"),
    auto_build: bool = typer.Option(False, "--auto-build", help="configure+build CPU llama.cpp libs if missing"),
) -> None:
    """Compile native/jlens_server against the vendored llama.cpp."""
    import os
    import subprocess

    from quant_tuner.paths import JLENS_SERVER_DIR

    env = dict(os.environ)
    if llama_dir is not None:
        env["LLAMA_CPP_DIR"] = str(llama_dir)
    cmd = ["bash", str(JLENS_SERVER_DIR / "build.sh")]
    if auto_build:
        cmd.append("--auto-build")
    rc = subprocess.call(cmd, env=env)
    raise typer.Exit(rc)


@lens_app.command("fit")
def fit(
    model: Path = typer.Option(..., help="model GGUF (F16 recommended; any quant works)"),
    corpus: Path = typer.Option(..., help="fitting corpus (text file or 'wikitext:N')"),
    out: Path = typer.Option(..., "-o", "--out", help="output lens GGUF"),
    layers: str | None = typer.Option(None, help="'a,b,c' or 'lo-hi' source layers (default: all)"),
    band_size: int = typer.Option(8, help="layers fit per pass (bounds Gram memory)"),
    ctx: int = typer.Option(4096, help="context length for the activation server"),
    max_seq_len: int = typer.Option(128, help="tokens per fitting prompt"),
    n_prompts: int = typer.Option(512, help="corpus prompts to fit on"),
    ridge: float = typer.Option(1e-3, help="ridge strength (relative to trace(Sxx)/d)"),
    ngl: int = typer.Option(99, help="GPU layers for the server"),
    gram_float32: bool = typer.Option(False, "--gram-float32", help="halve fit RAM (large models)"),
) -> None:
    """Fit a regression lens on MODEL over CORPUS (forward-only, any quant)."""
    import numpy as np

    from quant_tuner.lens.fitting import fit_lens

    layer_list = _parse_layers(layers)
    fit_lens(
        model, corpus, out, layers=layer_list, band_size=band_size, ctx=ctx,
        max_seq_len=max_seq_len, n_prompts=n_prompts, ridge=ridge, ngl=ngl,
        gram_dtype=np.float32 if gram_float32 else np.float64,
    )
    typer.echo(f"wrote {out}")


@lens_app.command("fit-causal")
def fit_causal(
    hf_model: Path = typer.Option(..., help="HF model dir (the F16 snapshot in ws.model_extracted)"),
    model: Path = typer.Option(..., help="matching model GGUF (for readout weights)"),
    corpus: Path = typer.Option(..., help="fitting corpus (text file)"),
    out: Path = typer.Option(..., "-o", "--out", help="output lens GGUF"),
    n_prompts: int = typer.Option(200, help="prompts (paper uses 1000x128tok; ~100 usable)"),
) -> None:
    """Fit the exact causal Jacobian via vendor/jacobian-lens, then convert to GGUF."""
    from quant_tuner.lens.causal import fit_causal_lens

    fit_causal_lens(hf_model, model, corpus, out, n_prompts=n_prompts)
    typer.echo(f"wrote {out}")


@lens_app.command("convert-pt")
def convert_pt(
    pt: Path = typer.Argument(..., help="reference lens .pt"),
    out: Path = typer.Argument(..., help="output lens GGUF"),
    base_model: str = typer.Option("", help="informational base-model tag"),
) -> None:
    """Convert a reference PyTorch lens (.pt) to the GGUF lens format (torch-free)."""
    from quant_tuner.lens.pt_convert import convert_pt_lens

    lens = convert_pt_lens(str(pt), str(out), base_model=base_model)
    typer.echo(f"converted {lens!r} -> {out}")


@lens_app.command("inspect")
def inspect(path: Path = typer.Argument(..., help="lens GGUF")) -> None:
    """Print lens GGUF metadata."""
    from quant_tuner.lens.lens import JacobianLensGGUF

    lens = JacobianLensGGUF.load(str(path))
    typer.echo(str(lens))
    typer.echo(f"  target_layer: {lens.target_layer}")
    typer.echo(f"  base_model:   {lens.base_model}")
    typer.echo(f"  biases:       {sorted(lens.biases) if lens.biases else 'none'}")
    if lens.extra_meta:
        typer.echo(f"  provenance:   {lens.extra_meta}")


@lens_app.command("capture")
def capture(
    model: Path = typer.Option(..., help="model GGUF to capture through"),
    runs_dir: Path = typer.Option(..., help="capture-run store directory"),
    prompt: str | None = typer.Option(None, help="prompt text"),
    prompt_file: Path | None = typer.Option(None, help="prompt text file"),
    traj: Path | None = typer.Option(None, help="swebench .traj.json to replay as a prompt"),
    layers: str | None = typer.Option(None, help="'a,b,c' or 'lo-hi' capture layers (default: all)"),
    tail: int = typer.Option(0, help="keep only the last N positions on disk (0 = all)"),
    n_predict: int = typer.Option(0, help="generate N tokens and include them"),
    tag: list[str] = typer.Option([], "--tag", help="k=v tags (repeatable)"),
    ctx: int = typer.Option(8192),
    ngl: int = typer.Option(99),
) -> None:
    """Capture a run (activations + metadata) — the unit of A/B diffing."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import capture_run
    from quant_tuner.lens.loops import load_trajectory_prompt

    tags = dict(t.split("=", 1) for t in tag if "=" in t)
    layer_list = _parse_layers(layers)
    with running_jlens_server(model, ctx=ctx, ngl=ngl) as client:
        if traj is not None:
            tokens = load_trajectory_prompt(traj, client)
        elif prompt is not None:
            tokens = client.tokenize(prompt, add_special=True, parse_special=True)
        elif prompt_file is not None:
            tokens = client.tokenize(prompt_file.read_text(), add_special=True, parse_special=True)
        else:
            raise typer.BadParameter("pass --prompt, --prompt-file, or --traj")
        keep = list(range(-tail, 0)) if tail > 0 else None
        run_dir = capture_run(
            client, tokens, runs_dir=runs_dir, layers=layer_list,
            keep_positions=keep, n_predict=n_predict,
            logits_positions=[-1], tags=tags,
        )
    typer.echo(f"run {run_dir.name} -> {run_dir}")


@lens_app.command("diff")
def diff(
    run_a: str = typer.Argument(..., help="candidate run id"),
    run_b: str = typer.Argument(..., help="reference run id (e.g. FP16)"),
    runs_dir: Path = typer.Option(..., help="capture-run store"),
    lens: Path = typer.Option(..., help="lens GGUF (shared by both runs)"),
    out: Path = typer.Option(..., help="output directory for diff.json + arrays"),
    pin: list[int] = typer.Option([], "--pin", help="pin token ids for rank trajectories"),
) -> None:
    """Diff two capture runs; write diff.json + npz + a summary."""
    from quant_tuner.lens.capture import load_run
    from quant_tuner.lens.diff import diff_runs
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    lens_obj = JacobianLensGGUF.load(str(lens))
    run = load_run(Path(runs_dir) / run_a)
    ref = load_run(Path(runs_dir) / run_b)
    readout_run = LensReadout(ReadoutWeights.from_gguf(run.manifest.model_path), lens_obj)
    readout_ref = LensReadout(ReadoutWeights.from_gguf(ref.manifest.model_path), lens_obj)
    d = diff_runs(run, ref, readout_run=readout_run, readout_ref=readout_ref,
                  pinned_tokens=list(pin) or None)
    d.save(out)
    for k, v in d.summary().items():
        typer.echo(f"  {k}: {v:.4f}")
    typer.echo(f"wrote {out}")


@lens_app.command("serve")
def serve(
    model: Path = typer.Option(..., help="model GGUF (backend A)"),
    lens: Path | None = typer.Option(None, help="lens GGUF (default: identity/logit lens)"),
    model_b: Path | None = typer.Option(None, help="second model GGUF for live A/B"),
    lens_b: Path | None = typer.Option(None, help="lens for backend B (default: same as --lens)"),
    runs_dir: Path | None = typer.Option(None, help="capture-run store for the run manager"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8090),
    ctx: int = typer.Option(8192),
    ngl: int = typer.Option(99),
) -> None:
    """Launch the interactive visualizer (with optional A/B backend)."""
    from quant_tuner.lens.server import serve_app

    serve_app(
        model=model, lens_path=lens, model_b=model_b, lens_b=lens_b,
        runs_dir=runs_dir, host=host, port=port, ctx=ctx, ngl=ngl,
    )


@lens_app.command("replay-toolcalls")
def replay_toolcalls(
    model: Path = typer.Option(..., help="quant GGUF to inspect"),
    lens: Path = typer.Option(..., help="lens GGUF (fit on F16)"),
    holdout: Path = typer.Option(..., help="tool-call holdout JSONL"),
    runs_dir: Path = typer.Option(..., help="capture-run store"),
    reference_runs: Path | None = typer.Option(None, help="FP16 reference run store for KLD/divergence"),
    csv: Path | None = typer.Option(None, help="leaderboard sidecar CSV to write"),
    decisions_out: Path | None = typer.Option(None, help="per-decision JSONL to write"),
    max_sessions: int | None = typer.Option(None),
    tail: int = typer.Option(16, help="decision-tail positions to capture"),
    ctx: int = typer.Option(8192),
    ngl: int = typer.Option(99),
) -> None:
    """Replay tool-call decisions through the lens; write a sidecar + per-decision log."""
    from quant_tuner.lens.replay import (
        replay_toolcall_lens,
        write_decisions_jsonl,
        write_lens_sidecar,
    )

    summary = replay_toolcall_lens(
        holdout, model, lens, runs_dir=runs_dir, reference_runs_dir=reference_runs,
        tail_positions=tail, max_sessions=max_sessions, ctx=ctx, ngl=ngl,
    )
    typer.echo(
        f"{summary.model}: {summary.n_decisions} decisions, "
        f"gold rank {summary.gold_tool_rank_final_mean:.2f}, "
        f"emergence layer {summary.emergence_layer_mean:.1f}"
    )
    if csv is not None:
        write_lens_sidecar([summary], csv)
        typer.echo(f"wrote sidecar {csv}")
    if decisions_out is not None:
        write_decisions_jsonl(summary, decisions_out)


@lens_app.command("loop")
def loop(
    model: Path = typer.Option(..., help="quant GGUF that loops"),
    lens: Path = typer.Option(..., help="lens GGUF"),
    traj: Path | None = typer.Option(None, help="swebench .traj.json to replay"),
    prompt_file: Path | None = typer.Option(None, help="prompt text file"),
    reference: Path | None = typer.Option(None, help="FP16 GGUF for collapse-depth comparison"),
    runs_dir: Path = typer.Option(..., help="capture-run store"),
    out: Path = typer.Option(..., help="output directory for the loop report"),
    sweep: bool = typer.Option(False, "--sweep", help="try ablation interventions to break the loop"),
    n_predict: int = typer.Option(512),
    ctx: int = typer.Option(8192),
    ngl: int = typer.Option(99),
) -> None:
    """Autopsy an infinite loop; optionally sweep ablations to break it."""
    import json

    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import load_run
    from quant_tuner.lens.loops import (
        intervention_sweep,
        load_trajectory_prompt,
        loop_autopsy,
        loop_direction,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with running_jlens_server(model, ctx=ctx, ngl=ngl) as client:
        if traj is not None:
            tokens = load_trajectory_prompt(traj, client)
        elif prompt_file is not None:
            tokens = client.tokenize(prompt_file.read_text(), add_special=True, parse_special=True)
        else:
            raise typer.BadParameter("pass --traj or --prompt-file")
    report = loop_autopsy(
        model, lens, tokens, runs_dir=runs_dir, reference_gguf=reference,
        n_predict=n_predict, ctx=ctx, ngl=ngl,
    )
    (out / "loop_report.json").write_text(json.dumps({
        "looped": report.looped, "notes": report.notes,
        "collapse_depth_quant": report.collapse_depth_quant,
        "collapse_depth_ref": report.collapse_depth_ref,
        "generated_text": report.generated_text[:2000],
    }, indent=1))
    typer.echo(report.notes)
    if not report.looped:
        return

    if sweep and report.quant_run_id is not None and report.loop is not None:
        run = load_run(Path(runs_dir) / report.quant_run_id)
        n_layer = len(run.manifest.capture_layers)
        late = run.manifest.capture_layers[int(n_layer * 2 / 3):]
        direction = loop_direction(run, report.loop, late[len(late) // 2])
        with running_jlens_server(model, ctx=ctx, ngl=ngl) as client:
            results = intervention_sweep(
                client, direction, layers=late, prompt_tokens=tokens,
                alphas=(0.5, 1.0, 1.5), n_predict=256,
            )
        winner = next((r for r in results if r.broke_loop), None)
        if winner is not None:
            direction.save(out / "direction.npz")
            typer.echo(f"loop broken by ablation a={winner.alpha} across {len(late)} layers")
            typer.echo(f"saved direction -> {out / 'direction.npz'}")
        else:
            typer.echo("no ablation in the sweep broke the loop")


@lens_app.command("probe")
def probe(
    model: Path = typer.Option(..., help="quant GGUF"),
    lens: Path = typer.Option(..., help="lens GGUF"),
    probes: Path = typer.Option(..., help="probe JSONL (id/prompt/gold/domain)"),
    reference: Path | None = typer.Option(None, help="FP16 GGUF for transition comparison"),
    out: Path = typer.Option(..., help="output directory"),
    ctx: int = typer.Option(8192),
    ngl: int = typer.Option(99),
) -> None:
    """Classify probes as correct / suppressed / absent; compare quant vs FP16."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.probes import (
        compare_probes,
        load_probes,
        render_probe_report,
        run_probes,
        write_probe_results,
    )
    from quant_tuner.lens.readout import LensReadout

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lens_obj = JacobianLensGGUF.load(str(lens))
    probe_list = load_probes(probes)

    q_readout = LensReadout(ReadoutWeights.from_gguf(str(model)), lens_obj)
    with running_jlens_server(model, ctx=ctx, ngl=ngl) as client:
        q_results = run_probes(client, q_readout, probe_list)
    write_probe_results(q_results, out / "quant_probes.jsonl")

    if reference is not None:
        r_readout = LensReadout(ReadoutWeights.from_gguf(str(reference)), lens_obj)
        with running_jlens_server(reference, ctx=ctx, ngl=ngl) as client:
            r_results = run_probes(client, r_readout, probe_list)
        write_probe_results(r_results, out / "ref_probes.jsonl")
        cmp = compare_probes(q_results, r_results)
        cmp.quant_model = Path(model).name
        cmp.ref_model = Path(reference).name
        render_probe_report(cmp, out / "probe_report.md")
        for k, v in cmp.summary().items():
            typer.echo(f"  {k}: {v}")


@lens_app.command("study")
def study(
    config: Path = typer.Option(..., help="study YAML (f16/quants/imatrices/lens/eval_corpora)"),
    out: Path = typer.Option(..., help="output directory"),
) -> None:
    """Run the 'why calibration works' divergence + weight-noise study."""
    from quant_tuner.lens.study import StudyConfig, calibration_study

    calibration_study(StudyConfig.from_yaml(config), out)
    typer.echo(f"wrote study to {out}")


@lens_app.command("bake")
def bake(
    f16: Path = typer.Option(..., help="F16 GGUF to edit"),
    direction: Path = typer.Option(..., help="direction.npz from a loop sweep"),
    quant_type: str = typer.Option(..., help="target quant type (e.g. IQ2_M)"),
    out_dir: Path = typer.Option(..., help="output directory"),
    imatrix: Path | None = typer.Option(None, help="imatrix GGUF for requantization"),
    layers: str | None = typer.Option(None, help="override layers to edit (default: direction's)"),
    verify_against: Path | None = typer.Option(None, help="original quant to verify KLD against"),
    reference: Path | None = typer.Option(None, help="FP16 GGUF baseline for verify KLD"),
    eval_dataset: Path | None = typer.Option(None, "--eval", help="eval corpus for verify KLD"),
) -> None:
    """Orthogonalize the direction out of F16, requantize, and verify."""
    from quant_tuner.lens.bake import Direction, bake_and_requantize, verify_bake

    baked = bake_and_requantize(
        f16, Direction.load(direction), quant_type=quant_type,
        imatrix=imatrix, out_dir=out_dir, layers=_parse_layers(layers),
    )
    typer.echo(f"baked -> {baked}")
    if verify_against is not None and reference is not None and eval_dataset is not None:
        result = verify_bake(
            baked, verify_against, reference=reference,
            eval_dataset=eval_dataset, out_dir=Path(out_dir) / "verify",
        )
        import json as _json

        typer.echo(_json.dumps(result, indent=2))


@lens_app.command("report")
def report(
    runs_dir: Path = typer.Option(..., help="capture-run store"),
    out: Path = typer.Option(..., help="output HTML report"),
) -> None:
    """Assemble an HTML report from capture runs and diffs."""
    from quant_tuner.lens.report import build_report

    build_report(runs_dir, out)
    typer.echo(f"wrote {out}")


def _parse_layers(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]
