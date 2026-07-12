"""Assemble a static HTML report from capture runs and diffs.

Self-contained (inline CSS, no external assets): a table of the capture runs
under a store, and — for any ``diff*/diff.json`` artifacts found — the scalar
divergence summaries side by side. Enough to skim an A/B batch without the
live UI; the interactive views live in the ``serve`` visualizer.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from quant_tuner.lens.capture import find_runs

_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a1a}
h1,h2{font-weight:600} table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-size:13px}
th{background:#f5f5f5} code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}
.tag{color:#666} tr:nth-child(even){background:#fafafa}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}
th{background:#222}td,th{border-color:#333}tr:nth-child(even){background:#181818}
code{background:#222}}
"""


def build_report(runs_dir: Path, out_html: Path) -> Path:
    runs_dir = Path(runs_dir)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    manifests = find_runs(runs_dir)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>J-Lens report — {html.escape(runs_dir.name)}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>J-Lens report</h1><p class='tag'>runs under <code>{html.escape(str(runs_dir))}</code></p>",
        "<h2>Capture runs</h2>",
        "<table><tr><th>run id</th><th>model</th><th>layers</th><th>positions</th>"
        "<th>gen</th><th>llama</th><th>tags</th></tr>",
    ]
    for m in manifests:
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(m.run_id)}</code></td>"
            f"<td>{html.escape(Path(m.model_path).name)}</td>"
            f"<td>{len(m.capture_layers)}</td>"
            f"<td>{len(m.positions)}</td>"
            f"<td>{m.n_gen}</td>"
            f"<td class='tag'>{html.escape(str(m.llama_commit))}</td>"
            f"<td class='tag'>{html.escape(json.dumps(m.tags))}</td>"
            "</tr>"
        )
    parts.append("</table>")

    diffs = sorted(runs_dir.glob("**/diff.json"))
    if diffs:
        parts.append("<h2>Diffs</h2>")
        parts.append(
            "<table><tr><th>candidate</th><th>reference</th>"
            "<th>agree</th><th>agree (final)</th><th>ref-top1 rank</th>"
            "<th>layer KLD</th><th>final KLD</th><th>first divergent layer</th></tr>"
        )
        for dj in diffs:
            meta = json.loads(dj.read_text())
            s = meta.get("summary", {})
            parts.append(
                "<tr>"
                f"<td>{html.escape(meta.get('run_model', ''))}</td>"
                f"<td>{html.escape(meta.get('ref_model', ''))}</td>"
                f"<td>{s.get('lens_agree_mean', 0):.3f}</td>"
                f"<td>{s.get('lens_agree_final_layer', 0):.3f}</td>"
                f"<td>{s.get('lens_ref_top1_rank_mean', 0):.2f}</td>"
                f"<td>{s.get('lens_layer_kld_auc', 0):.4f}</td>"
                f"<td>{s.get('lens_final_kld_mean', 0):.4f}</td>"
                f"<td>{s.get('lens_first_divergent_layer', 0):.0f}</td>"
                "</tr>"
            )
        parts.append("</table>")

    out_html.write_text("\n".join(parts) + "\n")
    return out_html
