// A/B diff heatmap rendering for quant-tuner's Jacobian-lens visualizer.
// Renders a layer x position grid over a decoded RunDiff, in three modes:
//   disagree  — orange where the candidate's lens top-1 != the reference's
//   rankshift — signed rank of the reference's top-1 token in the candidate
//   kld       — per-position final KLD strip (exact rows outlined)
// Self-contained (no d3 dependency): draws to a <canvas>.
'use strict';

function b64ToI32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int32Array(bytes.buffer);
}
function b64ToU8(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// diverging color for a signed, log-scaled rank shift (0 = agree)
function rankColor(rank) {
  if (rank <= 0) return '#f4f4f8';
  const t = Math.min(1, Math.log10(1 + rank) / 4); // rank 10000 -> 1
  const r = Math.round(245 - 130 * t);
  const g = Math.round(158 - 120 * t);
  const b = Math.round(11 + 40 * t);
  return `rgb(${r},${g},${b})`;
}

function renderDiffHeatmap(container, diff, mode) {
  const [L, P] = diff.shape;
  const agree = b64ToU8(diff.agree);            // L*P uint8
  const rankShift = b64ToI32(diff.ref_top1_rank_in_run); // L*P int32
  const CW = 16, CH = 12, PAD_L = 46, PAD_T = 18;
  const canvas = document.createElement('canvas');
  canvas.width = PAD_L + P * CW;
  canvas.height = PAD_T + L * CH + 34;
  const ctx = canvas.getContext('2d');
  ctx.font = '10px system-ui';
  ctx.fillStyle = '#666';
  ctx.fillText('layer', 2, PAD_T - 6);

  for (let li = 0; li < L; li++) {
    ctx.fillStyle = '#888';
    ctx.fillText(String(diff.meta.layers[li]), 2, PAD_T + li * CH + CH - 2);
    for (let p = 0; p < P; p++) {
      const idx = li * P + p;
      let color = '#f4f4f8';
      if (mode === 'disagree') {
        color = agree[idx] ? '#eef7ee' : '#f59e0b';
      } else if (mode === 'rankshift') {
        color = rankColor(rankShift[idx]);
      }
      ctx.fillStyle = color;
      ctx.fillRect(PAD_L + p * CW, PAD_T + li * CH, CW - 1, CH - 1);
    }
  }

  // per-position final-KLD strip under the grid
  const kld = diff.final_kld || [];
  const exact = diff.final_kld_exact || [];
  const kmax = Math.max(1e-6, ...kld);
  const stripY = PAD_T + L * CH + 6;
  ctx.fillStyle = '#666';
  ctx.fillText('KLD', 2, stripY + 10);
  for (let p = 0; p < P; p++) {
    const t = Math.min(1, (kld[p] || 0) / kmax);
    ctx.fillStyle = `rgb(${Math.round(80 + 175 * t)},${Math.round(80 - 60 * t)},${Math.round(120 - 60 * t)})`;
    ctx.fillRect(PAD_L + p * CW, stripY, CW - 1, 14);
    if (exact[p]) { // outline exact rows
      ctx.strokeStyle = '#111';
      ctx.strokeRect(PAD_L + p * CW + 0.5, stripY + 0.5, CW - 2, 13);
    }
  }

  container.innerHTML = '';
  const legend = document.createElement('div');
  legend.className = 'diff-legend';
  legend.innerHTML = mode === 'disagree'
    ? '<span style="color:#f59e0b">■</span> top-1 disagree &nbsp; <span style="color:#7ab77a">■</span> agree'
    : mode === 'rankshift'
      ? 'darker = reference top-1 fell further in the candidate'
      : 'strip: per-position final KLD (outlined = exact from stored logits)';
  container.appendChild(canvas);
  container.appendChild(legend);
}

window.JLensDiff = { renderDiffHeatmap, b64ToI32, b64ToU8 };
