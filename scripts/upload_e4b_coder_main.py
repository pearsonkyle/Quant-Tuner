#!/usr/bin/env python3
"""Publish the merged bf16 model, its card, its figures and its resume state.

Layout in the repo:

    /                       merged bf16 model -- what people load
    README.md               card + the generated evaluation appendix
    figures/                the appendix's images, referenced relatively
    checkpoints/step-7000/  adapter + optimizer + scheduler + rng + trainer state

The resume state is the point of the `checkpoints/` subtree: an adapter alone
cannot continue a run. Without `optimizer.pt` the Adam moments restart from
zero, which at this point in training is a worse perturbation than it sounds --
so the optimizer, scheduler, RNG state and trainer_state all ship alongside it.

Private by default. Verification comes before publicity: the GGUFs have to load
and the W4A16 has to serve before any of these repos go public.

    python scripts/upload_e4b_coder_main.py --dry-run     # list what would go
    python scripts/upload_e4b_coder_main.py
"""
import argparse
import os
from pathlib import Path

REPO = "pearsonkyle/gemma4-e4b-coder"
MERGED = Path("/workspace/models/gemma4-e4b-coder-merged")
CKPT = Path("/workspace/models/gemma4-e4b-stage1-131k-v65536/checkpoint-7000")
FIGS = Path("/workspace/LLM-Training-Kit/docs/figures")

# The model files people load. Listed explicitly rather than globbed so a stray
# file in the directory cannot be published by accident.
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"]

# Everything needed to resume. adapter_model.safetensors alone is NOT enough.
CKPT_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
              "scheduler.pt", "rng_state.pth", "trainer_state.json",
              "training_args.bin"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", required=True, help="assembled card + appendix")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true",
                    help="create public. Default is private; flip it by hand "
                         "once the artifacts are verified.")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    plan = []
    for f in MODEL_FILES:
        p = MERGED / f
        if not p.exists():
            raise SystemExit(f"missing model file: {p}")
        plan.append((p, f))
    for f in CKPT_FILES:
        p = CKPT / f
        if not p.exists():
            raise SystemExit(f"missing resume file: {p}\nThe point of this upload "
                             f"is that training can be continued; do not publish "
                             f"a partial checkpoint.")
        plan.append((p, f"checkpoints/step-7000/{f}"))
    for p in sorted(FIGS.glob("*.png")):
        plan.append((p, f"figures/{p.name}"))
    readme = Path(a.readme)
    if not readme.exists():
        raise SystemExit(f"missing README: {readme}")
    plan.append((readme, "README.md"))

    total = sum(p.stat().st_size for p, _ in plan)
    print(f"{len(plan)} files, {total/2**30:.2f} GiB -> {a.repo} "
          f"({'public' if a.public else 'private'})\n")
    for p, dest in plan:
        print(f"  {p.stat().st_size/2**20:10.1f} MiB  {dest}")
    if a.dry_run:
        print("\n(dry run -- nothing uploaded)")
        return

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    print(f"\nrepo ready: {a.repo}")

    for p, dest in plan:
        print(f"uploading {dest} ({p.stat().st_size/2**20:.1f} MiB)", flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                        repo_id=a.repo, repo_type="model")
    print("\nUPLOAD DONE")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    raise SystemExit(main())
