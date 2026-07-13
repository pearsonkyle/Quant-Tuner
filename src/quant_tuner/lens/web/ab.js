// A/B run manager + diff view for quant-tuner's Jacobian-lens visualizer.
// Adds a collapsible panel (toggled from the topbar) that:
//   - lists cached capture runs (/api/runs), filterable by tag
//   - captures the current prompt to backend A or B (/api/capture)
//   - picks two runs and renders their diff (/api/diff) via diff_heatmap.js
// Kept separate from app.js so the base single-model visualizer is unchanged;
// this activates only when the bridge reports a run store and/or backend B.
'use strict';

(function () {
  async function abApi(path, body) {
    const opts = body === undefined ? {} :
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    const r = await fetch(path, opts);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }

  const state = { backends: null, runs: [], selA: null, selB: null, mode: 'disagree' };

  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k === 'onclick') e.onclick = v;
      else if (k === 'html') e.innerHTML = v;
      else e.setAttribute(k, v);
    }
    for (const kid of kids) e.append(kid);
    return e;
  }

  async function refreshRuns(filter) {
    const q = filter ? ('?' + filter) : '';
    const data = await abApi('/api/runs' + q);
    state.runs = data.runs || [];
    renderRunList();
  }

  function renderRunList() {
    const list = document.getElementById('abRunList');
    if (!list) return;
    list.innerHTML = '';
    if (!state.runs.length) {
      list.append(el('div', { class: 'ab-empty' }, 'No capture runs yet. Capture the current prompt below.'));
      return;
    }
    for (const run of state.runs) {
      const tagStr = Object.entries(run.tags || {}).map(([k, v]) => `${k}=${v}`).join(' ');
      const row = el('div', { class: 'ab-run' },
        el('code', {}, run.run_id),
        el('span', { class: 'ab-run-model' }, run.model),
        el('span', { class: 'ab-run-tags' }, tagStr),
        el('button', { class: state.selA === run.run_id ? 'primary' : '', onclick: () => { state.selA = run.run_id; renderRunList(); } }, 'A'),
        el('button', { class: state.selB === run.run_id ? 'primary' : '', onclick: () => { state.selB = run.run_id; renderRunList(); } }, 'B'),
      );
      list.append(row);
    }
    const diffBtn = document.getElementById('abDiffBtn');
    if (diffBtn) diffBtn.disabled = !(state.selA && state.selB);
  }

  async function captureCurrent(backend) {
    const prompt = document.getElementById('prompt');
    const body = {
      backend,
      prompt: prompt ? prompt.value : '',
      tail: 32,
      logits_positions: [-1],
      tags: { via: 'ui', backend },
    };
    setAbStatus('capturing on backend ' + backend + '…');
    try {
      const res = await abApi('/api/capture', body);
      setAbStatus('captured ' + res.run_id);
      await refreshRuns();
    } catch (e) {
      setAbStatus('capture failed: ' + e.message, true);
    }
  }

  async function runDiff() {
    if (!state.selA || !state.selB) return;
    setAbStatus('computing diff…');
    try {
      const diff = await abApi(`/api/diff?a=${state.selA}&b=${state.selB}`);
      setAbStatus(`diff: agree ${(diff.meta.summary.lens_agree_mean * 100).toFixed(1)}%, ` +
        `first divergent layer ${diff.meta.summary.lens_first_divergent_layer}`);
      window.JLensDiff.renderDiffHeatmap(document.getElementById('abDiffView'), diff, state.mode);
    } catch (e) {
      setAbStatus('diff failed: ' + e.message, true);
    }
  }

  function setAbStatus(msg, isErr) {
    const el = document.getElementById('abStatus');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = isErr ? 'var(--danger)' : 'var(--dim)';
  }

  function buildPanel() {
    const hasB = state.backends && state.backends.b;
    const panel = el('div', { class: 'ab-panel', id: 'abPanel', style: 'display:none' },
      el('div', { class: 'ab-head' },
        el('b', {}, 'A/B run compare'),
        el('span', { class: 'ab-close', onclick: toggle }, '×')),
      el('div', { class: 'ab-cap' },
        el('span', {}, 'capture current prompt to: '),
        el('button', { onclick: () => captureCurrent('a') }, 'A: ' + (state.backends?.a?.model_name || 'model A')),
        hasB ? el('button', { onclick: () => captureCurrent('b') }, 'B: ' + state.backends.b.model_name) : el('span', {})),
      el('div', { class: 'ab-filter' },
        el('button', { onclick: () => refreshRuns('') }, '↻ all runs'),
        el('button', { onclick: () => refreshRuns('role=reference') }, 'references')),
      el('div', { class: 'ab-runs', id: 'abRunList' }),
      el('div', { class: 'ab-modes' },
        el('span', {}, 'mode: '),
        ...['disagree', 'rankshift', 'kld'].map(m =>
          el('button', { class: state.mode === m ? 'primary' : '', onclick: () => { state.mode = m; setMode(m); runDiff(); } }, m)),
        el('button', { class: 'primary', id: 'abDiffBtn', disabled: 'true', onclick: runDiff }, 'diff A↔B')),
      el('div', { class: 'ab-diffview', id: 'abDiffView' }),
      el('div', { class: 'ab-status', id: 'abStatus' }));
    document.body.append(panel);
  }

  function setMode(m) {
    document.querySelectorAll('.ab-modes button').forEach(b => {
      if (['disagree', 'rankshift', 'kld'].includes(b.textContent)) b.classList.toggle('primary', b.textContent === m);
    });
  }

  function toggle() {
    const p = document.getElementById('abPanel');
    if (!p) return;
    const showing = p.style.display !== 'none';
    p.style.display = showing ? 'none' : 'block';
    if (!showing) refreshRuns('');
  }

  async function init() {
    try {
      state.backends = await abApi('/api/backends');
    } catch (e) {
      return; // base UI (no A/B bridge) — silently do nothing
    }
    if (!state.backends.runs_dir && !state.backends.b) return;
    // add a topbar toggle button
    const bar = document.querySelector('.topbar');
    if (bar) {
      const btn = el('button', { class: 'ab-toggle', onclick: toggle }, 'A/B');
      btn.title = 'Compare two capture runs (FP16 vs quant, quant vs quant)';
      bar.append(btn);
    }
    buildPanel();
    // honor ?runA=&runB=&mode= URL params
    const u = new URLSearchParams(location.search);
    if (u.get('runA')) state.selA = u.get('runA');
    if (u.get('runB')) state.selB = u.get('runB');
    if (u.get('mode')) state.mode = u.get('mode');
    if (u.get('runA') || u.get('runB')) { toggle(); if (state.selA && state.selB) runDiff(); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
