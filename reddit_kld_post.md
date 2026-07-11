# how to make 2bit quants usable for agents

TL;DR: 2-bit quants of agentic models are usually junk, but the *reason* they're
junk isn't the bit width, it's the calibration. I compared three ways to build
the same 2-bit model (no calibration, imatrix, and AWQ) and actually benchmarked
them on a real tool-calling task instead of trusting KLD. The naive build is dead
on arrival, imatrix brings it back to life, and AWQ makes it genuinely usable.
And the static metrics (KLD/PPL/top_p) would have picked the wrong build.

## what "2-bit" actually means

A weight normally takes 16 bits. At "2-bit" you're keeping about 2-3 bits per
weight, so you're throwing away ~85% of the precision. The entire game is *which*
precision you keep. There are two families of 2-bit in llama.cpp:

- **Q2_K** ("K-quant"): blocks of weights share a scale, plain rounding. Runs
  with zero calibration. This is the naive baseline.
- **IQ2_M** ("i-quant"): weights snap to a shared non-uniform codebook (an
  E8 lattice). It literally *cannot run* without an importance matrix.

So the moment you go below ~3 bits, calibration stops being optional. The
question is just how good your calibration is.

## the three ways to calibrate

**1. Nothing (Q2_K).** Round everything uniformly. Fast, dumb, and at 2-bit it
falls apart.

**2. imatrix (importance matrix).** Run the model over real data and measure, per
input channel, how much that channel drives each layer's output. Hand that to the
quantizer so it spends its limited precision on the channels that matter. Nothing
about the weights changes, you're just telling the rounder where to be careful.

**3. AWQ (activation-aware weight quantization).** Also measures activations, but
instead of only *ranking* channels it *rescales* them: channels that carry big
activations get scaled up before quantizing (so they land on finer levels), and
the inverse scale folds into the previous norm layer so the fp16 math is
identical. Different lever. imatrix picks what to protect, AWQ reshapes the
weights so the important stuff survives rounding. It's still a plain GGUF with
zero extra inference cost.

I calibrate imatrix and AWQ on real coding/tool-use logs (the distribution the
model will actually see as an agent), not wikitext.

## why KLD / PPL / top_p isn't the full story

Those are the numbers everyone quotes, and they all measure the same thing:
how close the quant's output sits to fp16, *averaged over a text corpus*.

- **PPL**: how surprised the model is on average.
- **KLD**: distribution distance per token.
- **top_p / same-top**: how often the top token still matches fp16.

Here's the catch: those averages are dominated by common, easy tokens. The tokens
that actually decide an agent's behavior (the one that emits the tool call, the
argument values inside it) are rare, so they barely move the average. At 4-bit and
up everything tracks and it doesn't matter. At 2-bit those rare high-stakes tokens
are exactly what breaks first, and an average-token metric is blind to it.

So I ran an actual agentic eval: replay 25 held-out real tool-use sessions through
each quant (disjoint from calibration), and score per assistant turn:

- **tool-sel**: did it pick the right tool
- **param-acc**: did it fill in the right arguments
- **schema-valid**: did it even emit a well-formed call

## the benchmarks

Same eval and fp16 baseline for the static side, same 25-session replay for the
agentic side. Static: lower PPL/KLD and higher top_p is better. Agentic: higher
is better.

### gemma-4-31B (IQ2_M-class, ~3 bpw)

| build | PPL | KLD med | top_p | tool-sel | param-acc | schema-valid |
|---|---:|---:|---:|---:|---:|---:|
| Q2_K (no calibration) | 3523 | 5.21 | 25.4% | 0.000 | 0.000 | 0.000 |
| IQ2_M + imatrix | 1959 | **1.57** | **46.6%** | 0.454 | 0.171 | 0.805 |
| IQ2_M + AWQ | **1040** | 1.80 | 43.9% | **0.492** | **0.263** | **0.823** |

### Ornith-1.0-9B

| build | PPL | KLD med | top_p | tool-sel | param-acc | schema-valid |
|---|---:|---:|---:|---:|---:|---:|
| Q2_K (no calibration) | 54.3 | 2.03 | 37.9% | 0.026 | 0.000 | 0.026 |
| IQ2_M + imatrix | 6.4 | **0.11** | **80.6%** | 0.306 | 0.054 | 0.851 |
| IQ2_M + AWQ | 6.7 | 0.12 | 79.7% | **0.536** | **0.333** | **0.930** |

Two things jump out.

**Calibration is the whole ballgame at 2-bit.** The no-calibration Q2_K builds are
dead on arrival. gemma's scores a flat **0.000 on all three agentic metrics** (it
never once emitted a valid tool call across 3 reps), and the 9B is basically the
same (2.6% valid, 0% arguments right). Both are actually *bigger* on disk than the
IQ2_M builds and still useless. imatrix takes them from dead to functional. This
is exactly why the IQ2_* formats refuse to run without an imatrix in the first
place.

**Now look at imatrix vs AWQ, and look at the static columns.** On gemma, imatrix
has the *better* median KLD (1.57 vs 1.80) and better top_p. If you ranked by the
usual metrics you'd ship imatrix. But AWQ fills correct tool arguments 54% more
often (0.17 to 0.26 on gemma, 0.05 to 0.33 on the 9B) and is far more consistent
run to run. The static table points at the wrong build.

(Side note: on gemma AWQ actually *halves* PPL, 1959 to 1040, while KLD goes
slightly up. PPL and KLD disagree because the fold shifts the whole distribution,
lowering average surprise while nudging the single top token. Yet another reason
not to trust one static number.)

## the takeaway

- Below ~3 bits, calibration isn't optional, it's the difference between a dead
  model and a working one.
- imatrix and AWQ are different tools. imatrix ranks what to protect, AWQ reshapes
  the weights. For agentic 2-bit, AWQ preserved the tool-argument tokens better on
  every model I tried.
- KLD/PPL/top_p measure average fidelity, not task ability. At 2-bit they can and
  do point at the wrong build.
- **Benchmark the task you actually care about.** If you're running agents, replay
  real tool-use sessions and score arguments. It takes an afternoon and it changed
  which quant I shipped.

Everything here is plain GGUF, runs in stock llama.cpp / Ollama / LM Studio, no
custom runtime. The AWQ scales fold into the norms so inference cost is identical
to the imatrix build at the same bit width.
