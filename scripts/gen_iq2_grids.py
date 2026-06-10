"""Regenerate src/quant_tuner/calibrate/_iq2_grids.py from llama.cpp sources.

The IQ2_* quants snap each group of 8 weights to an entry of a fixed
E8-lattice codebook (``iq2xxs_grid`` / ``iq2xs_grid`` / ``iq2s_grid`` in
ggml-common.h). The AWQ low-bit proxies need those exact tables; this script
parses them out of the vendored llama.cpp checkout and emits a standalone
Python module so the package works without the submodule present.

Run after bumping the llama.cpp pin:

    uv run python scripts/gen_iq2_grids.py \
        --source vendor/llama.cpp/ggml/src/ggml-common.h \
        --commit $(git -C vendor/llama.cpp rev-parse HEAD)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

GRIDS = {
    "iq2xxs": 256,
    "iq2xs": 512,
    "iq2s": 1024,
}

OUT_PATH = Path(__file__).resolve().parents[1] / "src" / "quant_tuner" / "calibrate" / "_iq2_grids.py"


def parse_table(text: str, name: str, count: int) -> list[int]:
    m = re.search(
        rf"GGML_TABLE_BEGIN\(uint64_t,\s*{name}_grid,\s*{count}\)(.*?)GGML_TABLE_END\(\)",
        text,
        re.DOTALL,
    )
    if not m:
        raise SystemExit(f"table {name}_grid[{count}] not found")
    vals = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]{16}", m.group(1))]
    if len(vals) != count:
        raise SystemExit(f"{name}_grid: expected {count} entries, parsed {len(vals)}")
    return vals


def parse_ksigns(text: str) -> list[int]:
    m = re.search(
        r"GGML_TABLE_BEGIN\(uint8_t,\s*ksigns_iq2xs,\s*128\)(.*?)GGML_TABLE_END\(\)",
        text,
        re.DOTALL,
    )
    if not m:
        raise SystemExit("ksigns_iq2xs table not found")
    vals = [int(tok) for tok in re.findall(r"\d+", m.group(1))]
    if len(vals) != 128:
        raise SystemExit(f"ksigns_iq2xs: expected 128 entries, parsed {len(vals)}")
    return vals


def entry_component_bytes(v: int) -> bytes:
    """uint64 -> the 8 magnitude bytes in component order (j-th = byte j, LE)."""
    return bytes((v >> (8 * j)) & 0xFF for j in range(8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="path to llama.cpp's ggml/src/ggml-common.h")
    ap.add_argument("--commit", default="unknown",
                    help="llama.cpp commit hash, recorded in the header")
    args = ap.parse_args()

    text = args.source.read_text()

    # Sanity-check the sign convention the proxies hard-code: every entry of
    # ksigns_iq2xs has an even number of set bits (= even number of negative
    # signs per group of 8 is the representable set for IQ2_XXS / IQ2_XS).
    ksigns = parse_ksigns(text)
    assert all(bin(v).count("1") % 2 == 0 for v in ksigns), \
        "ksigns parity convention changed — update the IQ2 proxies"

    blocks = []
    for name, count in GRIDS.items():
        vals = parse_table(text, name, count)
        comp = b"".join(entry_component_bytes(v) for v in vals)
        magnitudes = sorted(set(comp))
        hex_lines = "\n".join(
            f'    "{comp[i:i + 64].hex()}"' for i in range(0, len(comp), 64)
        )
        blocks.append(
            f"# {name}_grid: {count} entries x 8 components, "
            f"magnitudes {magnitudes}\n"
            f"_{name.upper()}_HEX = (\n{hex_lines}\n)\n"
        )

    OUT_PATH.write_text(
        '"""E8-lattice codebooks for the IQ2_* quants — GENERATED, do not edit.\n'
        "\n"
        f"Extracted from llama.cpp ggml-common.h at commit {args.commit}\n"
        "by scripts/gen_iq2_grids.py. Each grid entry is 8 magnitude bytes\n"
        "(component order); signs live outside the codebook. For IQ2_XXS and\n"
        "IQ2_XS only sign patterns with an EVEN number of negatives per group\n"
        "of 8 are representable (ksigns_iq2xs parity); IQ2_S/IQ2_M store free\n"
        "sign bytes.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        + "\n".join(blocks)
        + "\n"
        "_HEX = {\n"
        + "".join(f'    "{n}": _{n.upper()}_HEX,\n' for n in GRIDS)
        + "}\n"
        "\n"
        "\n"
        "def grid_components(name: str) -> bytes:\n"
        '    """Raw codebook for ``name`` ("iq2xxs"/"iq2xs"/"iq2s"): K*8 magnitude bytes."""\n'
        '    return bytes.fromhex("".join(_HEX[name]))\n'
    )
    n_bytes = OUT_PATH.stat().st_size
    print(f"wrote {OUT_PATH} ({n_bytes} bytes)")


if __name__ == "__main__":
    main()
