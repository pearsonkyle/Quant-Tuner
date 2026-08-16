#!/usr/bin/env python3
"""Vision smoke test for a served multimodal checkpoint.

Mirrors the check the GGUF ladder shipped: a synthetic image with three shapes whose
colour AND position are both unambiguous, so a pass requires the vision tower to have
survived quantization rather than the language model guessing plausible words.

Why synthetic rather than a photo: a real photo can be described correctly from priors
("a dog on grass") without the encoder contributing much. Three coloured shapes at three
named positions cannot — getting shape+colour+position right for all three is only
possible if the encoder works. Position is the part that degrades first.

    PYTHONPATH=src .venv/bin/python scripts/vision_smoke.py --base-url http://127.0.0.1:18080/v1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request


def build_image() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([60, 170, 220, 330], fill="red")                      # left
    d.rectangle([420, 180, 590, 300], fill="blue")                  # right
    d.polygon([(320, 460), (250, 350), (390, 350)], fill="green")   # bottom centre
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask(base_url: str, model: str, png: bytes, prompt: str, max_tokens: int) -> str:
    b64 = base64.b64encode(png).decode()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{base_url}/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default=None, help="write result JSON here")
    a = ap.parse_args()

    png = build_image()
    print(f"test image: 640x480, red circle LEFT / blue rectangle RIGHT / green triangle BOTTOM-CENTRE "
          f"({len(png)} bytes PNG)\n")

    try:
        answer = ask(a.base_url.rstrip("/"), a.model, png,
                     "Describe this image. For each shape state its colour, its shape, "
                     "and where it is positioned.", a.max_tokens)
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        print("\nA 400 mentioning image/content parts usually means the served checkpoint "
              "has no vision tower — check that the export was built with "
              "--model-class Qwen3_5ForConditionalGeneration.")
        return 1

    print("--- model answer ---")
    print(answer)
    print("--------------------\n")

    low = answer.lower()
    # Each shape scores only if colour, shape AND a correct position word all appear.
    checks = {
        "red circle on the left":       all(w in low for w in ("red",)) and
                                        any(w in low for w in ("circle", "ellipse")) and
                                        any(w in low for w in ("left",)),
        "blue rectangle on the right":  "blue" in low and
                                        any(w in low for w in ("rectangle", "square", "rect")) and
                                        "right" in low,
        "green triangle bottom-centre": "green" in low and "triangle" in low and
                                        any(w in low for w in ("bottom", "below", "lower")),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n = sum(checks.values())
    verdict = ("VISION OK" if n == 3 else
               "PARTIAL — encoder present but degraded" if n else
               "VISION BROKEN — no shape identified")
    print(f"\n{n}/3 -> {verdict}")

    if a.out:
        json.dump({"answer": answer, "checks": checks, "score": n, "verdict": verdict},
                  open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0 if n == 3 else 2


if __name__ == "__main__":
    sys.exit(main())
