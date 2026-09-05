"""An OpenAI-shaped client that runs a local Gemma 4 checkpoint in-process.

`quant_tuner.eval.toolcall` only ever touches `client.chat.completions.create`
and reads `.choices[0].message`, so it does not need a socket. Skipping HTTP is
not a shortcut here, it avoids three specific problems:

* `flash_attn` 2.8.3 is ABI-pinned to the torch 2.9.1 in the training venv. A
  separate serving venv gets a different torch and loses FA2 -- and without FA2
  the 35 sliding-attention layers fall back to sdpa, which cannot express a
  sliding window and does full quadratic work. That is what OOMs at 131K.
* No new packages land in the venv a 7-day training run is executing from.
* The prompt is rendered by the checkpoint's own `chat_template.jinja`, so the
  eval sees exactly the string training saw. An HTTP server in between is one
  more place for that to drift, and it has drifted before: the first bpb
  harness silently dropped every tool response and scored 6% of each
  conversation.

Responses are real `openai` types, not stand-ins, so anything that accepts an
`OpenAI` client accepts this.

    from quant_tuner.eval.local_gemma4 import LocalGemma4Client
    client = LocalGemma4Client(BASE, adapter=CKPT, max_len=131072)
    run_toolcall_eval(client=client, ...)
"""

from __future__ import annotations

import sys
import time
from typing import Any

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from quant_tuner.eval.gemma4_wire import (
    deserialize_tool_arguments,
    parse_generation,
)

LLMTK = "/workspace/LLM-Training-Kit"


class LocalGemma4Client:
    def __init__(self, base: str, adapter: str | None = None, *,
                 device: str = "cuda", max_len: int = 131072,
                 dtype: str = "bfloat16", route_attention: bool = True,
                 verbose: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_len = max_len
        self.device = device
        self.verbose = verbose
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=getattr(torch, dtype), device_map=device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

        self.routed: list[int] = []
        if route_attention and device != "cpu":
            # Per-layer routing: FA2 on the 35 sliding layers, sdpa on the 7
            # full-attention ones (head_dim 512 exceeds FA2's 256 cap). Without
            # this a 131K prefill does not fit. Best-effort: an eval that has to
            # run on sdpa is slower and shorter, not wrong.
            try:
                sys.path.insert(0, LLMTK)
                from llmtk.llm import mixed_attn
                self.routed = mixed_attn.apply(self.model)
            except Exception as e:  # noqa: BLE001
                print(f"[local_gemma4] attention routing unavailable ({e}); "
                      "using the model's default", flush=True)
        if verbose:
            tag = f"+{adapter.rstrip('/').split('/')[-1]}" if adapter else "base"
            print(f"[local_gemma4] {tag} loaded in {time.time()-t0:.0f}s"
                  f"{f', sdpa on layers {self.routed}' if self.routed else ''}",
                  flush=True)
        self.chat = _Chat(self)

    # ------------------------------------------------------------------
    def _generate(self, messages: list[dict], tools: list[dict] | None,
                  temperature: float, max_tokens: int, top_p: float | None,
                  seed: int | None, **_: Any) -> tuple[dict, int, int]:
        import torch

        msgs = deserialize_tool_arguments(messages)
        text = self.tok.apply_chat_template(
            msgs, tools=tools or None, add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)
        n_in = ids["input_ids"].shape[-1]
        if n_in > self.max_len:
            # Head-truncate, matching `long_example_strategy: truncate_head`:
            # keep the system prompt and leading turns, drop the tail.
            for k in ids:
                ids[k] = ids[k][:, :self.max_len]
            n_in = self.max_len
        ids = {k: v.to(self.model.device) for k, v in ids.items()}

        if seed is not None:
            torch.manual_seed(seed)
        greedy = not temperature or temperature <= 0
        kw: dict[str, Any] = {"do_sample": not greedy, "max_new_tokens": max_tokens}
        if not greedy:
            kw["temperature"] = temperature
            if top_p is not None:
                kw["top_p"] = top_p
        with torch.no_grad():
            out = self.model.generate(**ids, **kw)
        new = out[0][n_in:]
        gen = self.tok.decode(new, skip_special_tokens=False)
        return parse_generation(gen), n_in, int(new.shape[-1])


    # ------------------------------------------------------------------
    def generate_batch(self, requests: list[dict], *, temperature: float = 0.0,
                       max_tokens: int = 1536, top_p: float | None = None,
                       batch_size: int = 8) -> list[dict]:
        """Run several independent prompts through one `generate` call.

        Decode, not prefill, is what a generation eval spends its time on, and
        decode is memory-bandwidth bound: the weights are re-read for every
        token regardless of how many sequences are riding along. Batching eight
        prompts therefore costs barely more wall-clock than one.

        Only worth it when prompts are of comparable length -- padding is waste,
        and a batch runs until its LONGEST member finishes. Uniform sets (the
        MTG decision points) batch well; the tool-call sessions, whose prefixes
        span 400 to 87,000 tokens, do not, which is why the shared turn-replay
        harness still goes one at a time through `create`.

        Each request is `{"messages": [...], "tools": [...]}`; returns parsed
        dicts in the same order.
        """
        import torch

        pad = self.tok.pad_token_id
        if pad is None:
            pad = self.tok.eos_token_id
        prev_side = self.tok.padding_side
        self.tok.padding_side = "left"   # decoder-only: pad on the left
        out: list[dict] = []
        try:
            for start in range(0, len(requests), batch_size):
                chunk = requests[start:start + batch_size]
                texts = [
                    self.tok.apply_chat_template(
                        deserialize_tool_arguments(r["messages"]),
                        tools=r.get("tools") or None,
                        add_generation_prompt=True, tokenize=False)
                    for r in chunk
                ]
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len,
                               add_special_tokens=False)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                n_in = enc["input_ids"].shape[-1]
                greedy = not temperature or temperature <= 0
                kw = {"do_sample": not greedy, "max_new_tokens": max_tokens,
                      "pad_token_id": pad}
                if not greedy:
                    kw["temperature"] = temperature
                    if top_p is not None:
                        kw["top_p"] = top_p
                with torch.no_grad():
                    gen = self.model.generate(**enc, **kw)
                for j in range(len(chunk)):
                    new = gen[j][n_in:]
                    txt = self.tok.decode(new, skip_special_tokens=False)
                    parsed = parse_generation(txt)
                    parsed["_n_out"] = int((new != pad).sum())
                    parsed["_max_tokens"] = max_tokens
                    out.append(parsed)
                if self.verbose:
                    print(f"[local_gemma4] batch {start//batch_size+1}: "
                          f"{len(chunk)} prompts, {n_in} ctx", flush=True)
        finally:
            self.tok.padding_side = prev_side
        return out


class _Chat:
    def __init__(self, owner: LocalGemma4Client):
        self.completions = _Completions(owner)


class _Completions:
    def __init__(self, owner: LocalGemma4Client):
        self.o = owner

    def create(self, *, model: str = "local", messages: list[dict],
               tools: list[dict] | None = None, temperature: float = 0.0,
               max_tokens: int = 512, top_p: float | None = None,
               seed: int | None = None, **kw: Any) -> ChatCompletion:
        parsed, n_in, n_out = self.o._generate(
            messages, tools, temperature, max_tokens, top_p, seed, **kw)
        tcs = [
            ChatCompletionMessageToolCall(
                id=c["id"], type="function",
                function=Function(name=c["function"]["name"],
                                  arguments=c["function"]["arguments"]),
            )
            for c in parsed["tool_calls"]
        ] or None
        msg = ChatCompletionMessage(
            role="assistant", content=parsed["content"], tool_calls=tcs)
        # `length` matters to the caller: a truncated generation is not a
        # refusal to call a tool, and scoring the two the same hides a
        # max_tokens that is set too low.
        truncated = parsed.get("truncated_thought") or n_out >= max_tokens
        finish = "tool_calls" if tcs else ("length" if truncated else "stop")
        return ChatCompletion(
            id=f"local-{int(time.time()*1000)}", object="chat.completion",
            created=int(time.time()), model=model,
            choices=[Choice(index=0, finish_reason=finish, message=msg)],
            usage=CompletionUsage(prompt_tokens=n_in, completion_tokens=n_out,
                                  total_tokens=n_in + n_out),
        )
