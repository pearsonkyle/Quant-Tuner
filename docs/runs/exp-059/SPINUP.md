# SPINUP — fresh coder-QAT run from new trajectory data

The complete path from a new dataset JSONL to a benchmarked Q2_0, incorporating every
exp-059 lesson. On the current box (95 GiB CUDA card, 1.5 TB RAM, 384 cores) the
wall-clock is: corpus ~10 min, KD table ~9.5 h, training ~55-65 h, eval ~1 h.

## 0. Prereqs (verify, don't assume)

- Student HF weights: out/exp-057/model (the ONLY local copy; prism chat template).
- KD teacher: SWE-Lego/SWE-Lego-Qwen3-32B in $HF_HOME (62 GB). Tokenizer-compatible
  (padded 151,936 vs 151,669 is fine; kd_precompute verifies).
- vendor/llama.cpp-prism build (Q2_0 = ftype 41 is fork-only): llama-server, llama-quantize.
- SWE-mimic harness: tools/swe_mimic in-repo; live copy at /workspace/swe-mimic with
  .venv (openai-agents) + standing work/ instances (dask eval + 6 harvest).
- anchor10 inputs: out/exp-058/kd/{rep_bank.json,rep_traj_contexts.jsonl,teacher_probe_32b.json}
  (also tracked in docs/runs/exp-058/inputs/).
- Disk ≥150 G free. Campaign artifacts that eat it: latents 28 G x ckpt-keep,
  KD table ~10 G, export intermediates ~20-50 G per bench (benchwatch prunes; the one
  time it didn't, disk hit 100% and an export failed mid-campaign).

## 1. Dataset intake gates (all were load-bearing in exp-059)

Rows: {id, source, split, messages[], tools[], meta{instance_id, repo, language,
resolved,…}} — OpenAI chat format, reasoning_content on assistant turns.

```bash
# a. schema/split/token census + group-clean split (0 instances straddling train/test)
# b. eval disjointness: instance_ids vs out/external/swe-rebench/{holdout,holdout50,
#    harvest_instances,qat_train_instances}.jsonl + the 7 mimic instances  → MUST be 0
# c. PATH CENSUS — count checkout-root values across tool-call args. If any root >~20%
#    of conversations, diversify:
python scripts/path_diversify_sft.py --inp NEW.jsonl.gz --out NEW_pathdiv.jsonl.gz
#    (deterministic per instance_id; 12% keep-share; verify census after)
# d. resolved-only? note it: it teaches never-conclude (exp-059 law 6). If mixing in
#    conclude-when-stuck examples, this is the place.
```

## 2. Corpus (window 32768 / max-tool-tokens 8192 / min-density 0.05)

```bash
# merge with the anchor10 universal sources (broad-instruct + refusals guard behavior)
# then:
PYTHONPATH=src .venv/bin/python scripts/build_sft_qat_corpus.py \
  --sft out/corpora/<name>/sft.jsonl.gz --window 32768 --max-tool-tokens 8192 \
  --min-density 0.05 --out out/exp-0XX/corpus_<tag>_32768.pt          # + --split test for val
PYTHONPATH=src .venv/bin/python scripts/inspect_corpus_window.py <corpus.pt> --audit
PYTHONPATH=src .venv/bin/python scripts/analyze_stop_context.py <corpus.pt>
# read the build log per source: reasoning survival, tool-truncation %, /testbed counts
```

## 3. KD table (32B forced-stop, AFTER the corpus is frozen)

```bash
PYTHONPATH=src .venv/bin/python scripts/kd_precompute.py \
  --teacher SWE-Lego/SWE-Lego-Qwen3-32B --corpus <corpus.pt> \
  --include-ids 151645 --out out/exp-0XX/kd/<tag>_32b_topk64_fs151645.pt
# ~11.2 s/window on this card; check coverage≈0.998 at trainer startup
```

## 4. Train — the coder3 configuration (the settled one)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QAT_LOGIT_CHUNK=512
# GRAD_ACCUM: target ~600 optimizer steps → accum = round(n_windows / 600)  (coder3: 2974→4)
TAG=<tag> GRAD_ACCUM=4 LR=5e-4 STEER=0.1 CLIP=0.25 REP=0.1 REP_CAP=0.6 \
REP_K=1,2,3,4,5 REP_N=10 \
REP_BANK=out/exp-058/kd/rep_bank.json \
REP_TRAJ=out/exp-058/kd/rep_traj_contexts.jsonl \
CORPUS=<corpus.pt> VAL=<val.pt> TABLE=<kd.pt> \
TEACHER_PROBE=out/exp-058/kd/teacher_probe_32b.json \
OUT=out/exp-0XX/kd32b-full-<tag> EPOCHS=1.0 \
bash scripts/run_kd_anchor_qat.sh
```
Expect: ~313 s/step at accum 4 (incl. GradOffload transfers), peak 91.4-91.6 GiB,
val descending from ~0.8, flips pacing to ~2-3%/run, occasional PROBE-WARN that the
steering hinge corrects within one probe interval (strike 2 aborts, by design).

## 5. Watchers (start immediately after launch)

```bash
# report.html every 10 min:
(until [ -f $OUT/train.log ]; do sleep 60; done; sleep 120; \
 exec bash scripts/qat_report_watch.sh $OUT 600) &
# agentic bench every ~200 accum-steps (THE instrument that caught coder1 AND coder2):
RUN=$OUT PREFIX=<tag> EVERY=200 EPISODES=3 bash scripts/qat_benchwatch.sh > benchwatch.log &
```
Read benchwatch lines against the coder3 baselines: s400 12%-unique/streak-46 was
FAILING; s600 56%-unique/streak-11/first-edits was RECOVERING. testbed_cmds>0 with a
diversified corpus, or out_tok pinned at the cap with steps≤2, means stop the run.

## 6. Eval the artifact

```bash
VAL_CORPUS=<val.pt> bash scripts/run_kd_export_bench.sh <tag> $OUT/trained_latents.pt
# then ALWAYS re-run the mimic at serving temperature (the chain's default T=0.25
# reproduces the sharpening loop on every model incl. anchor10):
TEMP=0.7 LABEL=<TAG>-T07-Q2_0 bash scripts/run_swe_mimic.sh <tag>   # x3
```
Serve at temperature 0.7, top_p 0.95, NO repeat/presence penalties.

## Known failure signatures (fastest diagnosis)

| symptom | cause | fix |
|---|---|---|
| OOM at launch, GBs reserved-unallocated | allocator fragmentation | expandable_segments |
| OOM in ternary STE at accum>1, ~0 reserved | grads resident across micro-batches | --accum-offload auto is default; verify the "accum grad offload" line printed |
| episodes fixate on one path vs explicit prompt | env-constant monoculture in data | path census + path_diversify_sft.py |
| mute episodes, out_tok at cap, probe textbook | total drift >> validated envelope | check flips %; raise GRAD_ACCUM (never lower LR below ~4e-4) |
| mid-run ckpt GGUF probes hot + mute episode | export caught a steering transient | check train.log PROBE-WARN/recovery; judge the next probe-clean ckpt |
| every episode loops at eval | benched at T=0.25 | T=0.7, no penalties |
| EXPORT FAILED in benchwatch | disk full (intermediates) | prune_export_intermediates.sh; keep ≥150 G free |
