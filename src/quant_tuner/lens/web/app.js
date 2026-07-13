// jlens-gguf interactive visualizer.
// Heatmap / panels / rank charts adapted from Anthropic's jacobian-lens
// slice page (Apache-2.0); rewired for a live bridge API with interventions.
'use strict';

// ---------------------------------------------------------------------------
// utilities
// ---------------------------------------------------------------------------

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const COLORS = ['#e15759','#4e79a7','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac','#f28e2b'];

async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function b64ToI32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int32Array(bytes.buffer);
}

function setStatus(msg, isErr) {
  const el = $('status');
  el.textContent = msg || '';
  el.classList.toggle('err', !!isErr);
}

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------

let PROPS = null;          // /api/props
let VOCAB = null;          // array of piece strings
let D = null;              // current slice: {ctxId, tokens, pieces, nPrompt, nGen, layers, topN, topIds, norms, useLens}
let baselineByKey = new Map();   // tokensKey -> {argmax: Int32Array[T*L]}
let baselinePending = null;

let selCtx = 0, selLayer = 0;
let showWs = false;
let pinned = new Map();          // tid -> color
let activeTid = null;
let rankCache = new Map();       // tid -> Int32Array[T*L]  (per current ctx)
let rankFetching = new Set();
let interventions = [];          // [{id, type, enabled, ...params}]
let ivCounter = 0;
let running = false;

const at = (arr, t, l, k) => arr[(t * D.L + l) * D.topN + k];
const argmaxAt = (t, l) => at(D.topIds, t, l, 0);
const layerIdx = () => D.layerIndex.get(selLayer) ?? 0;
const tokStr = tid => (VOCAB && VOCAB[tid] !== undefined) ? VOCAB[tid] : `[${tid}]`;

function clean(tid) {
  let s = tokStr(tid);
  if (s === '') return '·';
  let t = s.replace(/\n/g, '⏎');
  t = showWs ? t.replace(/ /g, '␣') : t.replace(/^\s+|\s+$/g, '');
  if (!t) t = '·';
  return t;
}

function nextColor() {
  const used = new Set(pinned.values());
  return COLORS.find(c => !used.has(c)) || COLORS[pinned.size % COLORS.length];
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

(async function boot() {
  try {
    PROPS = await api('/api/props');
    const lens = PROPS.lens;
    const lensBadge = lens.method === 'identity'
      ? `<span class="badge lens-id" title="No fitted lens loaded: readouts use the raw logit lens (J = I). Fit one with: python -m jlens_gguf fit">logit lens</span>`
      : `<span class="badge lens-fit" title="Fitted over ${lens.n_prompts} prompts; layers ${lens.source_layers[0]}–${lens.source_layers[lens.source_layers.length-1]}">${esc(lens.method)} lens · ${lens.source_layers.length} layers</span>`;
    $('modelInfo').innerHTML =
      `<b>${esc(PROPS.model_name)}</b> · ${PROPS.n_layers}L · d=${PROPS.d_model} · vocab ${PROPS.n_vocab.toLocaleString()}${lensBadge}`;
    setStatus('loading vocab…');
    VOCAB = (await api('/api/vocab')).pieces;
    setStatus('');
  } catch (e) {
    $('modelInfo').textContent = 'bridge unreachable: ' + e.message;
    setStatus(String(e.message), true);
    return;
  }
  $('runBtn').onclick = () => runSlice();
  $('prompt').addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runSlice(); }
  });
  $('useLens').onchange = () => { if (D) runSlice(); };
  $('addSteer').onclick = () => openIvModal('steer');
  $('addSwap').onclick = () => openIvModal('swap');
  $('addAblate').onclick = () => openIvModal('ablate');
  const params = new URLSearchParams(location.search);
  if (params.get('prompt')) $('prompt').value = params.get('prompt');
  else if (!$('prompt').value) $('prompt').value = 'Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is';
  if (params.has('gen')) $('nPredict').value = +params.get('gen') || 0;
  if (params.has('autorun')) runSlice();
})();

// ---------------------------------------------------------------------------
// slice running
// ---------------------------------------------------------------------------

function activeSpecs() {
  return interventions.filter(iv => iv.enabled).map(iv => {
    if (iv.type === 'steer') return { type: 'steer', token_id: iv.token, alpha: iv.alpha, layers: iv.layers, pos: iv.pos };
    if (iv.type === 'swap') return { type: 'swap', token_a: iv.tokenA, token_b: iv.tokenB, layers: iv.layers, pos: iv.pos };
    return { type: 'ablate', token_id: iv.token, layers: iv.layers, pos: iv.pos };
  });
}

async function runSlice(extra = {}) {
  if (running) return;
  running = true;
  $('runBtn').disabled = true;
  setStatus('running…');
  try {
    const body = {
      use_lens: $('useLens').checked,
      stride: +$('stride').value || 1,
      n_predict: +$('nPredict').value || 0,
      interventions: activeSpecs(),
      ...extra,
    };
    if (!extra.tokens) {
      const prompt = $('prompt').value;
      if (!prompt.trim()) throw new Error('empty prompt');
      if ($('chatMode').checked) body.messages = [{ role: 'user', content: prompt }];
      else body.prompt = prompt;
    }
    const res = await api('/api/slice', body);
    ingestSlice(res);
    const t = res.timings;
    setStatus(`fwd ${Math.round(t.forward_ms)}ms · grid ${Math.round(t.grid_ms)}ms · ${res.tokens.length} pos × ${res.layers.length} layers` +
              (body.interventions.length ? ` · ${body.interventions.length} interventions` : ''));
    maybeFetchBaseline(res);
  } catch (e) {
    setStatus(String(e.message), true);
  } finally {
    running = false;
    $('runBtn').disabled = false;
  }
}

// Baseline identity = prompt tokens + which layers (encodes stride) + lens
// mode. A stride or lens change yields a different key, so the diff baseline is
// refetched instead of being silently compared against a mismatched grid.
function baselineKey(tokens, nPrompt, layers, useLens) {
  return tokens.slice(0, nPrompt).join(',') + '|' + (useLens ? 1 : 0) + '|' + layers.join('-');
}

function ingestSlice(res) {
  const firstRun = !D;
  D = {
    ctxId: res.ctx_id,
    tokens: res.tokens,
    pieces: res.pieces,
    nPrompt: res.n_prompt,
    nGen: res.n_gen,
    layers: res.layers,
    layerIndex: new Map(res.layers.map((l, i) => [l, i])),
    topN: res.top_n,
    T: res.tokens.length,
    L: res.layers.length,
    topIds: b64ToI32(res.top_ids),
    norms: res.norms,
    useLens: res.use_lens,
    vocabSize: res.vocab_size,
    hadInterventions: (res.interventions || []).length > 0,
  };
  rankCache = new Map();
  rankFetching = new Set();
  // record this run as baseline if it had no interventions
  if (!D.hadInterventions) {
    baselineByKey.set(baselineKey(D.tokens, D.nPrompt, D.layers, D.useLens), snapshotArgmax());
    if (baselineByKey.size > 6) baselineByKey.delete(baselineByKey.keys().next().value);
  }
  if (firstRun) buildLayout();
  else rebuildForNewData();
  // keep the current selection if it's still a valid position (note: 0 is
  // valid — don't fall through `||`); otherwise default to the last position
  selCtx = Number.isFinite(selCtx) ? Math.min(Math.max(selCtx, 0), D.T - 1) : D.T - 1;
  if (!D.layerIndex.has(selLayer)) selLayer = D.layers[Math.floor(D.L / 2)];
  if (firstRun) { selCtx = D.T - 1; }
  for (const tid of pinned.keys()) ensureRank(tid);
  render();
  hmScrollTo(selCtx);
  refreshCellPanel();
}

function snapshotArgmax() {
  // prompt positions only — generated tokens differ run-to-run and aren't
  // comparable, so they must not seed a diff baseline
  const nP = Math.min(D.nPrompt, D.T);
  const out = new Int32Array(nP * D.L);
  for (let t = 0; t < nP; t++) for (let l = 0; l < D.L; l++) out[t * D.L + l] = argmaxAt(t, l);
  return { argmax: out, T: nP, L: D.L };
}

function currentBaseline() {
  if (!D || !D.hadInterventions) return null;
  const base = baselineByKey.get(baselineKey(D.tokens, D.nPrompt, D.layers, D.useLens));
  return (base && base.L === D.L) ? base : null;
}

async function maybeFetchBaseline(res) {
  if (!(res.interventions || []).length) return;
  const key = baselineKey(res.tokens, res.n_prompt, res.layers, res.use_lens);
  if (baselineByKey.has(key) || baselinePending === key) return;
  baselinePending = key;
  try {
    const base = await api('/api/slice', {
      tokens: res.tokens.slice(0, res.n_prompt),
      use_lens: res.use_lens, top_n: res.top_n, interventions: [],
      stride: +$('stride').value || 1,
    });
    // only store if the baseline's grid matches the steered run's layers;
    // otherwise the stride control moved mid-fetch and this is stale
    if (base.layers.join('-') !== res.layers.join('-')) return;
    const bIds = b64ToI32(base.top_ids);
    const T = base.tokens.length, L = base.layers.length, K = base.top_n;
    const argmax = new Int32Array(T * L);
    for (let t = 0; t < T; t++) for (let l = 0; l < L; l++) argmax[t * L + l] = bIds[(t * L + l) * K];
    baselineByKey.set(key, { argmax, T, L });
    if (D && D.hadInterventions) { hmRepaint(true); }
    setStatus('baseline computed — changed cells marked ▲');
  } catch (e) { /* diff is best-effort */ }
  finally { baselinePending = null; }
}

// ---------------------------------------------------------------------------
// layout construction (once)
// ---------------------------------------------------------------------------

const tooltip = $('tt');
let hmWrap, hmCanvas, hmSel, hmHover, hmXInner, toksDiv;
let hmCtx2d, hmVisW = 0, hmVisH = 0, hmCol0 = -1, hmCol1 = -1, hmSx = -1;
const CELL_W = 56, CELL_H = 14, YAX_W = 30, XAX_H = 32;
let geomV = 0;
let byLayerDiv, byCtxDiv, byLayerRows = [];
let byCtxRows = [], byCtxR0 = -1, byCtxR1 = -1, byCtxSpacer, byCtxPool;
let tokSpans = [];

function buildLayout() {
  $('main').innerHTML = `
  <div class="column heatmap-column" id="col0">
    <div class="controls-bar">
      <span class="readout">Pos <b id="ctxLabel"></b><code id="ctxTok"></code> Layer <b id="layerLabel"></b></span>
      <label class="ws-row"><input type="checkbox" id="wsChk"> whitespace</label>
      <span class="hint">⇧ hover scrubs · ←→ pos · ↑↓ layer · click cell = select + pin</span>
    </div>
    <div class="heatmap-wrapper" id="hm-wrap"></div>
    <div class="v-resize" id="tokResize"></div>
    <div class="bottom-controls">
      <div class="tokens-area" id="tokens"></div>
      <div class="rank-row">
        <div class="rank-col">
          <div class="rank-title" id="rankTitle"></div>
          <div class="rank-heatmap" id="rankHm"></div>
        </div>
        <div class="pinned-row" id="pinned"></div>
      </div>
      <div class="meta-line" id="meta"></div>
    </div>
  </div>
  <div class="resize-handle" data-col="0"></div>
  <div class="column panel-column" id="col1">
    <h2 id="byLayerH"></h2><div class="token-rows" id="byLayer"></div>
    <div class="rank-chart" id="byLayerChart"></div>
  </div>
  <div class="resize-handle" data-col="1"></div>
  <div class="column panel-column" id="col2">
    <h2 id="byCtxH"></h2><div class="token-rows" id="byCtx"></div>
    <div class="rank-chart" id="byCtxChart"></div>
  </div>
  <div class="column tools-column" id="col3">
    <div class="tool-section">
      <h3><span>Cell readout</span><span class="muted" id="cellLoc"></span></h3>
      <div class="tool-body">
        <div class="cell-readout" id="cellReadout"><span class="muted">select a cell…</span></div>
        <div class="tool-actions">
          <button id="steerCellBtn" title="Steer with this cell's top token">steer top ▸</button>
          <button id="swapCellBtn" title="Swap this cell's top token with another">swap top ▸</button>
          <button id="decompBtn" title="Sparse J-space decomposition of this activation (matching pursuit over J-lens vectors)">decompose</button>
        </div>
        <div class="cell-readout" id="decompOut"></div>
      </div>
    </div>
    <div class="tool-section">
      <h3><span>Generate</span></h3>
      <div class="tool-body">
        <div class="frow" style="display:flex;gap:8px;align-items:center">
          <label class="muted">tokens</label><input type="number" id="genN" value="24" min="1" max="256">
          <label class="chk muted"><input type="checkbox" id="genGreedy" checked> greedy</label>
          <label class="muted">T</label><input type="number" id="genTemp" value="0.8" min="0" max="2" step="0.1" style="width:44px">
        </div>
        <div class="tool-actions">
          <button id="genBtn" class="primary">generate</button>
          <button id="genVizBtn" title="Run the slice with +N generated tokens so the grid extends into the continuation">visualize continuation</button>
        </div>
        <div id="genOut"></div>
      </div>
    </div>
    <div class="tool-section">
      <h3><span>Live backend</span><span class="muted" id="liveState">…</span></h3>
      <div class="tool-body">
        <div class="muted" style="margin-bottom:6px">OpenAI-compatible endpoint for apps:<br>
          <code id="oaiUrl" style="user-select:all"></code></div>
        <div class="tool-actions">
          <button id="livePush" title="Install the enabled intervention chips as the live set applied to every completion this backend serves (position ranges become all-positions)">push interventions ▸</button>
          <button id="liveClear" title="Remove all live interventions from the backend">clear</button>
          <button id="liveLoad" title="Load the backend's most recent completion (prompt + generated tokens) into the visualizer">load last chat ⤵</button>
        </div>
        <div class="muted" id="liveInfo" style="margin-top:6px"></div>
      </div>
    </div>
    <div class="tool-section">
      <h3><span>Lens</span></h3>
      <div class="tool-body muted" id="lensInfo"></div>
    </div>
  </div>`;

  hmWrap = $('hm-wrap');
  toksDiv = $('tokens');
  byLayerDiv = $('byLayer');
  byCtxDiv = $('byCtx');

  $('wsChk').onchange = () => { showWs = $('wsChk').checked; hmRepaint(true); refillPanels(); render(); };
  $('decompBtn').onclick = runDecompose;
  $('steerCellBtn').onclick = () => openIvModal('steer', { token: argmaxAt(selCtx, layerIdx()) });
  $('swapCellBtn').onclick = () => openIvModal('swap', { tokenA: argmaxAt(selCtx, layerIdx()) });
  $('genBtn').onclick = runGenerate;
  $('genVizBtn').onclick = () => { $('nPredict').value = +$('genN').value; runSlice(); };

  const lens = PROPS.lens;
  $('lensInfo').innerHTML = lens.method === 'identity'
    ? `No fitted lens — using the <b>logit lens</b> (J = I). Steering still works but uses raw unembedding directions.<br><br>Fit a lens:<br><code>python -m jlens_gguf fit --model … -o lens.gguf</code>`
    : `<b>${esc(lens.method)}</b> lens, ${lens.source_layers.length} fitted layers, target L${lens.target_layer}, fitted on ${lens.n_prompts} prompts.${lens.has_bias ? ' Affine (with bias).' : ''}`;

  $('livePush').onclick = livePush;
  $('liveClear').onclick = liveClear;
  $('liveLoad').onclick = liveLoad;
  pollLive();
  if (!window.__livePoll) window.__livePoll = setInterval(pollLive, 5000);

  buildHeatmapDom();
  buildPanels();
  wireKeyboardAndResize();
}

// ---------------------------------------------------------------------------
// live backend (interventions applied to the OpenAI endpoint apps talk to)
// ---------------------------------------------------------------------------

let lastSeenCompletion = 0;

async function pollLive() {
  try {
    const st = await api('/api/live/state');
    $('oaiUrl').textContent = st.openai_url;
    $('liveState').textContent = `${st.count} active`;
    const lc = st.last_completion;
    if (lc && lc.id) {
      const fresh = lc.id !== lastSeenCompletion ? ' · new' : '';
      $('liveInfo').textContent =
        `last chat: #${lc.id} · ${lc.n_prompt}+${lc.n_gen} tokens · ${lc.interventions_active} interventions${fresh}`;
    } else {
      $('liveInfo').textContent = 'no completions served yet — point your app at the endpoint above';
    }
  } catch (e) { $('liveState').textContent = 'offline'; }
}

async function livePush() {
  try {
    const res = await api('/api/live/push', { interventions: activeSpecs() });
    setStatus(`live set installed: ${res.count} native edits — the backend now steers every completion`);
    pollLive();
  } catch (e) { setStatus(String(e.message), true); }
}

async function liveClear() {
  try {
    await api('/api/live/clear', {});
    setStatus('live interventions cleared');
    pollLive();
  } catch (e) { setStatus(String(e.message), true); }
}

async function liveLoad() {
  try {
    const last = await api('/api/live/last');
    const st = await api('/api/live/state');
    lastSeenCompletion = st.last_completion.id;
    setStatus(`loading last backend chat (${last.tokens.length} tokens)…`);
    // replay with the backend's own live set, so readouts match what the app got
    await runSlice({ tokens: last.tokens, interventions: st.ui_specs });
    // restore the app's prompt/generation boundary (the slice sees all tokens
    // as prompt); rebuild so generated tokens are marked and the boundary line
    // is drawn
    if (D && last.n_gen > 0) {
      D.nPrompt = last.n_prompt;
      D.nGen = last.n_gen;
      buildHeatmapDom();
      render();
      selCtx = Math.min(D.nPrompt, D.T - 1);
      render();
      refreshCellPanel();
    }
  } catch (e) { setStatus(String(e.message), true); }
}

function rebuildForNewData() {
  buildHeatmapDom();
  buildPanels();
  geomV++;
}

// ---------------------------------------------------------------------------
// heatmap (virtualized canvas; adapted from reference)
// ---------------------------------------------------------------------------

function buildHeatmapDom() {
  const totalW = YAX_W + D.T * CELL_W, totalH = D.L * CELL_H + XAX_H;
  hmWrap.innerHTML = `
    <div class="hm-spacer" style="width:${totalW}px;height:${totalH}px">
      <canvas class="hm-canvas"></canvas>
      <div class="hm-sel" style="width:${CELL_W-2}px;height:${CELL_H-2}px"></div>
      <div class="hm-hover" style="width:${CELL_W-2}px;height:${CELL_H-2}px"></div>
      <div class="hm-yaxis" style="width:${YAX_W}px;margin-top:${-D.L*CELL_H}px">
        ${D.layers.map((l,i)=>`<div style="height:${CELL_H}px;line-height:${CELL_H}px">${D.layers[D.L-1-i]}</div>`).join('')}
      </div>
      <div class="hm-xaxis"><div class="hm-xaxis-inner"></div></div>
    </div>`;
  hmCanvas = hmWrap.querySelector('.hm-canvas');
  hmSel = hmWrap.querySelector('.hm-sel');
  hmHover = hmWrap.querySelector('.hm-hover');
  hmXInner = hmWrap.querySelector('.hm-xaxis-inner');

  hmWrap.onscroll = () => hmRepaint();
  hmWrap.onmousemove = e => {
    const c = hmCellAt(e.clientX, e.clientY);
    if (!c) { hmHover.style.display = 'none'; tooltip.classList.remove('visible'); return; }
    hmHover.style.display = 'block';
    hmHover.style.left = (YAX_W + c.t * CELL_W) + 'px';
    hmHover.style.top = ((D.L - 1 - c.li) * CELL_H) + 'px';
    if (e.shiftKey) { selCtx = c.t; selLayer = D.layers[c.li]; schedule(); }
    showTT(e, c.t, c.li);
  };
  hmWrap.onmouseleave = () => { hmHover.style.display = 'none'; tooltip.classList.remove('visible'); };
  hmWrap.onclick = e => {
    const c = hmCellAt(e.clientX, e.clientY);
    if (!c) return;
    selCtx = c.t; selLayer = D.layers[c.li];
    pinToken(argmaxAt(c.t, c.li));
    refreshCellPanel();
  };
  hmSizeCanvas();

  // tokens strip
  toksDiv.innerHTML = '';
  tokSpans = D.pieces.map((s, i) => {
    const sp = document.createElement('span');
    sp.className = 'token' + (i >= D.nPrompt ? ' gen' : '');
    sp.textContent = s;
    sp.onclick = () => { selCtx = i; render(); refreshCellPanel(); };
    sp.onmouseenter = e => { if (e.shiftKey) { selCtx = i; schedule(); } };
    toksDiv.appendChild(sp);
    return sp;
  });
  $('meta').textContent =
    `${D.T} positions × ${D.L} layers · top-${D.topN}` +
    (D.nGen ? ` · last ${D.nGen} generated` : '') +
    (D.useLens ? '' : ' · LOGIT LENS (lens off)');
}

function hmSizeCanvas() {
  hmVisW = hmWrap.clientWidth || 800; hmVisH = D.L * CELL_H;
  const dpr = devicePixelRatio || 1;
  hmCanvas.width = Math.round(hmVisW * dpr); hmCanvas.height = Math.round(hmVisH * dpr);
  hmCanvas.style.width = hmVisW + 'px'; hmCanvas.style.height = hmVisH + 'px';
  hmCtx2d = hmCanvas.getContext('2d'); hmCtx2d.scale(dpr, dpr);
  hmSx = -1;
}

function hmRepaint(force = false) {
  if (!D) return;
  const sx = hmWrap.scrollLeft;
  if (!force && sx === hmSx) return;
  hmSx = sx;
  const c0 = Math.max(0, Math.floor((sx - YAX_W) / CELL_W));
  const c1 = Math.min(D.T, Math.ceil((sx + hmVisW - YAX_W) / CELL_W));
  hmCol0 = c0; hmCol1 = c1;
  const base = currentBaseline();
  hmCtx2d.clearRect(0, 0, hmVisW, hmVisH);
  hmCtx2d.textBaseline = 'middle';
  for (let t = c0; t < c1; t++) {
    const x = YAX_W + t * CELL_W - sx;
    for (let li = D.L - 1; li >= 0; li--) {
      const y = (D.L - 1 - li) * CELL_H, tid = argmaxAt(t, li);
      const pin = pinned.get(tid);
      if (pin) { hmCtx2d.fillStyle = pin + '20'; hmCtx2d.fillRect(x, y, CELL_W, CELL_H); }
      hmCtx2d.strokeStyle = '#eee'; hmCtx2d.strokeRect(x + .5, y + .5, CELL_W - 1, CELL_H - 1);
      hmCtx2d.font = '9px monospace';
      hmCtx2d.fillStyle = pin || '#000';
      const txt = clean(tid);
      hmCtx2d.fillText(txt.length > 9 ? txt.slice(0, 8) + '…' : txt, x + 3, y + CELL_H / 2, CELL_W - 6);
      // diff marker: top-1 changed vs baseline (prompt positions only)
      if (base && t < base.T && base.argmax[t * D.L + li] !== tid) {
        hmCtx2d.fillStyle = '#f59e0b';
        hmCtx2d.beginPath();
        hmCtx2d.moveTo(x + CELL_W - 8, y + 1); hmCtx2d.lineTo(x + CELL_W - 1, y + 1);
        hmCtx2d.lineTo(x + CELL_W - 1, y + 8); hmCtx2d.closePath(); hmCtx2d.fill();
      }
    }
    // generation boundary
    if (D.nGen && t === D.nPrompt) {
      hmCtx2d.fillStyle = '#0d9488';
      hmCtx2d.fillRect(x - 1, 0, 2, D.L * CELL_H);
    }
  }
  hmXInner.style.left = (YAX_W + c0 * CELL_W) + 'px';
  hmXInner.innerHTML = Array.from({length: c1 - c0}, (_, i) => {
    const t = c0 + i, s = (D.pieces[t] || '').replace(/\n/g, '⏎').trim() || '·';
    return `<div class="hm-xaxis-col${t >= D.nPrompt ? ' gen' : ''}" style="width:${CELL_W}px"><span>${esc(s.slice(0,8))}</span><span>${t}</span></div>`;
  }).join('');
}

function hmPlaceSel() {
  hmSel.style.left = (YAX_W + selCtx * CELL_W) + 'px';
  hmSel.style.top = ((D.L - 1 - layerIdx()) * CELL_H) + 'px';
}

function hmCellAt(clientX, clientY) {
  const r = hmWrap.getBoundingClientRect();
  const t = Math.floor((clientX - r.left + hmWrap.scrollLeft - YAX_W) / CELL_W);
  const li = D.L - 1 - Math.floor((clientY - r.top + hmWrap.scrollTop) / CELL_H);
  return (t >= 0 && t < D.T && li >= 0 && li < D.L) ? { t, li } : null;
}

function hmScrollTo(t) {
  if (t >= hmCol0 && t < hmCol1 - 1) return;
  const x = YAX_W + t * CELL_W;
  if (x < hmWrap.scrollLeft + YAX_W) hmWrap.scrollLeft = x - YAX_W;
  else if (x + CELL_W > hmWrap.scrollLeft + hmVisW) hmWrap.scrollLeft = x + CELL_W - hmVisW;
}

function showTT(e, t, li) {
  const rows = [];
  for (let k = 0; k < D.topN; k++) {
    const tid = at(D.topIds, t, li, k);
    rows.push(`<tr><td class="tt-rank">${k}</td><td class="tt-token${pinned.has(tid) ? ' pinned' : ''}">${esc(clean(tid))}</td></tr>`);
  }
  const base = currentBaseline();
  const diff = base && t < base.T && base.argmax[t * D.L + li] !== argmaxAt(t, li)
    ? `<div style="color:#b45309">baseline: ${esc(clean(base.argmax[t * D.L + li]))}</div>` : '';
  tooltip.innerHTML = `<div class="tt-header">Pos ${t} · Layer ${D.layers[li]}</div>${diff}<table class="tt-table">${rows.join('')}</table>`;
  tooltip.classList.add('visible');
  const bb = tooltip.getBoundingClientRect();
  tooltip.style.left = Math.min(innerWidth - bb.width - 8, e.clientX + 12) + 'px';
  tooltip.style.top = Math.max(4, e.clientY - bb.height - 8) + 'px';
}

// ---------------------------------------------------------------------------
// side panels (byLayer / byCtx)
// ---------------------------------------------------------------------------

function buildPanels() {
  // byLayer: one row per layer (top of stack first)
  byLayerDiv.innerHTML = '';
  byLayerRows = [];
  for (let i = 0; i < D.L; i++) {
    const li = D.L - 1 - i;
    const r = document.createElement('div'); r.className = 'token-row';
    const lab = document.createElement('span'); lab.className = 'row-label'; lab.textContent = D.layers[li];
    r.appendChild(lab);
    const cells = [];
    for (let k = 0; k < D.topN; k++) {
      const c = document.createElement('span'); c.className = 'token-cell';
      c.onclick = e => { e.stopPropagation(); pinToken(+c.dataset.tid); };
      r.appendChild(c); cells.push(c);
    }
    r.onclick = () => { selLayer = D.layers[li]; render(); refreshCellPanel(); };
    r.onmouseenter = e => { if (e.shiftKey) { selLayer = D.layers[li]; schedule(); } };
    byLayerDiv.appendChild(r);
    byLayerRows.push({ r, cells, li });
  }
  // byCtx: virtualized rows over positions
  byCtxDiv.innerHTML = '';
  byCtxDiv.style.position = 'relative';
  byCtxSpacer = document.createElement('div');
  byCtxSpacer.style.height = (D.T * 15) + 'px';
  byCtxDiv.appendChild(byCtxSpacer);
  byCtxPool = document.createElement('div');
  byCtxPool.style.cssText = 'position:absolute;top:0;left:0;right:0';
  byCtxDiv.appendChild(byCtxPool);
  byCtxRows = []; byCtxR0 = -1; byCtxR1 = -1;
  byCtxDiv.onscroll = () => byCtxRefill(layerIdx(), false);
}

const ROW_H = 15;
function byCtxEnsurePool(n) {
  while (byCtxRows.length < n) {
    const r = document.createElement('div'); r.className = 'token-row'; r.style.height = ROW_H + 'px';
    const lab = document.createElement('span'); lab.className = 'row-label'; r.appendChild(lab);
    const ct = document.createElement('span'); ct.className = 'row-ctx-token'; r.appendChild(ct);
    const cells = [];
    for (let k = 0; k < D.topN; k++) {
      const c = document.createElement('span'); c.className = 'token-cell';
      c.onclick = e => { e.stopPropagation(); pinToken(+c.dataset.tid); };
      r.appendChild(c); cells.push(c);
    }
    r.onclick = () => { selCtx = +r.dataset.t; render(); refreshCellPanel(); };
    r.onmouseenter = e => { if (e.shiftKey) { selCtx = +r.dataset.t; schedule(); } };
    byCtxPool.appendChild(r); byCtxRows.push({ r, lab, ct, cells });
  }
}

function byCtxRefill(li, force) {
  const r0 = Math.max(0, Math.floor(byCtxDiv.scrollTop / ROW_H));
  const r1 = Math.min(D.T, r0 + Math.ceil(byCtxDiv.clientHeight / ROW_H) + 1);
  if (!force && r0 === byCtxR0 && r1 === byCtxR1) return;
  byCtxR0 = r0; byCtxR1 = r1;
  byCtxEnsurePool(r1 - r0);
  byCtxPool.style.transform = `translateY(${r0 * ROW_H}px)`;
  for (let i = 0; i < byCtxRows.length; i++) {
    const row = byCtxRows[i], t = r0 + i;
    if (t >= r1) { row.r.style.display = 'none'; continue; }
    row.r.style.display = '';
    row.r.dataset.t = t;
    row.r.classList.toggle('active-row', t === selCtx);
    row.lab.textContent = t;
    row.ct.textContent = (D.pieces[t] || '').replace(/\n/g, '⏎');
    fillRow(row.cells, t, li);
  }
}

function byCtxScrollTo(t) {
  if (t >= byCtxR0 && t < byCtxR1 - 1) return;
  const y = t * ROW_H;
  if (y < byCtxDiv.scrollTop) byCtxDiv.scrollTop = y;
  else if (y + ROW_H > byCtxDiv.scrollTop + byCtxDiv.clientHeight) byCtxDiv.scrollTop = y + ROW_H - byCtxDiv.clientHeight;
  byCtxRefill(layerIdx(), false);
}

function fillRow(cells, t, li) {
  for (let k = 0; k < D.topN; k++) {
    const tid = at(D.topIds, t, li, k), c = cells[k];
    if (+c.dataset.tid !== tid || c.__ws !== showWs) {
      c.dataset.tid = tid; c.__ws = showWs;
      c.textContent = clean(tid);
    }
    const pin = pinned.get(tid);
    c.style.cssText = pin ? `background:${pin}20;text-decoration:underline 2px ${pin};text-underline-offset:2px` : '';
  }
}

function refillPanels() { lastKey.byLayerP = lastKey.byCtxP = null; }

// ---------------------------------------------------------------------------
// pins + ranks
// ---------------------------------------------------------------------------

function rankAt(t, li, tid) {
  const arr = rankCache.get(tid);
  return arr ? arr[t * D.L + li] : -1;
}

async function ensureRank(tid) {
  if (rankCache.has(tid) || rankFetching.has(tid)) return;
  rankFetching.add(tid);
  try {
    const res = await api('/api/ranks', { ctx_id: D.ctxId, token_ids: [tid] });
    if (res.ctx_id === D.ctxId) {
      rankCache.set(tid, b64ToI32(res.ranks));
      render();
    }
  } catch (e) { /* stale ctx */ }
  finally { rankFetching.delete(tid); }
}

function pinToken(tid) {
  if (pinned.has(tid)) {
    pinned.delete(tid);
    if (activeTid === tid) activeTid = pinned.size ? [...pinned.keys()][0] : null;
  } else {
    pinned.set(tid, nextColor());
    activeTid = tid;
    ensureRank(tid);
  }
  refillPanels(); render();
}

// ---------------------------------------------------------------------------
// charts (adapted from reference)
// ---------------------------------------------------------------------------

function crispCanvas(parentSel, W, H, { overlay = false } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const sel = parentSel.append('canvas')
    .attr('width', Math.round(W * dpr)).attr('height', Math.round(H * dpr))
    .style('width', W + 'px').style('height', H + 'px');
  if (overlay) sel.style('position', 'absolute').style('top', 0).style('left', 0).style('pointer-events', 'none');
  const ctx = sel.node().getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

function ggFrame(g, x, y, w, h, { xTicks = 5, yTicks = 4, yFmt = '~s' } = {}) {
  g.append('rect').attr('width', w).attr('height', h).attr('fill', '#f4f4f4');
  const grid = g.append('g').attr('class', 'grid');
  grid.selectAll('.gx').data(x.ticks(xTicks)).join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d)).attr('y1', 0).attr('y2', h);
  grid.selectAll('.gy').data(y.ticks(yTicks)).join('line')
    .attr('y1', d => y(d)).attr('y2', d => y(d)).attr('x1', 0).attr('x2', w);
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(xTicks).tickSize(0).tickPadding(4));
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(yTicks, yFmt).tickSize(0).tickPadding(4));
}

function drawRankChart(div, xVals, xLabel, valueOf, onScrub) {
  const W = div.clientWidth || 240, H = div.clientHeight || 150;
  const m = { t: 12, r: 10, b: 26, l: 36 }, w = W - m.l - m.r, h = H - m.t - m.b;
  if (w < 30 || h < 30) { div.__updateScrub = null; return; }
  const VMAX = D.vocabSize || 50000;
  const useCanvas = xVals.length > 200;
  let st = div.__chart;
  if (!st || st.geomV !== geomV) {
    d3.select(div).selectAll('*').remove();
    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0, w]);
    const y = d3.scaleLog().domain([0.9, VMAX]).range([0, h]);
    const svg = d3.select(div).append('svg').attr('width', W).attr('height', H);
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
    ggFrame(g, x, y, w, h);
    g.append('text').attr('x', w / 2).attr('y', h + 22).attr('text-anchor', 'middle').attr('fill', '#999').attr('font-size', 9).text(xLabel + ' →');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('fill', '#999').attr('font-size', 9).text('rank');
    const dataG = g.append('g');
    let cv = null;
    if (useCanvas) { cv = crispCanvas(d3.select(div), W, H, { overlay: true }); cv.translate(m.l, m.t); }
    const scrub = g.append('line').attr('y1', 0).attr('y2', h).attr('stroke', '#999').attr('stroke-dasharray', '3,3').attr('pointer-events', 'none');
    const dots = g.append('g').attr('pointer-events', 'none');
    g.append('rect').attr('width', w).attr('height', h).attr('fill', 'transparent').style('cursor', 'pointer')
      .on('mousemove click', e => {
        if (e.type === 'mousemove' && !e.shiftKey) return;
        const raw = x.invert(d3.pointer(e)[0]);
        const nearest = xVals.reduce((a, b) => Math.abs(b - raw) < Math.abs(a - raw) ? b : a);
        onScrub(nearest); schedule();
      });
    st = div.__chart = { x, y, w, h, dataG, cv, scrub, dots, geomV, series: [] };
  }
  const { x, y, dataG, cv, scrub, dots } = st;
  if (pinned.size === 0) { dataG.selectAll('*').remove(); cv?.clearRect(0, 0, st.w, st.h); st.series = []; div.__updateScrub = null; return; }
  if (cv) cv.clearRect(0, 0, st.w, st.h);
  dataG.selectAll('*').remove();
  st.series = [];
  for (const [tid, color] of pinned) {
    const pts = xVals.map((xv, xi) => ({ x: xv, r: valueOf(xi, tid) })).filter(p => p.r >= 0);
    st.series.push({ tid, color, pts });
    if (!pts.length) continue;
    if (cv) {
      cv.fillStyle = color;
      for (const p of pts) cv.fillRect(x(p.x) - 1, y(p.r + 1) - 1, 2, 2);
    } else {
      dataG.append('path').datum(pts).attr('fill', 'none').attr('stroke', color)
        .attr('stroke-width', tid === activeTid ? 2.5 : 1.5)
        .attr('opacity', activeTid != null && tid !== activeTid ? 0.3 : 1)
        .attr('d', d3.line().x(p => x(p.x)).y(p => y(Math.max(1, p.r + 1))));
    }
  }
  div.__updateScrub = xv => {
    const px = x(xv); scrub.attr('x1', px).attr('x2', px);
    dots.selectAll('*').remove();
    for (const { color, pts } of st.series) {
      const p = pts.find(d => d.x === xv);
      if (p) dots.append('circle').attr('cx', px).attr('cy', y(p.r + 1)).attr('r', 4).attr('fill', color).attr('stroke', '#fff').attr('stroke-width', 1.5);
    }
  };
}

function drawRankHm() {
  const div = $('rankHm');
  d3.select(div).selectAll('*').remove();
  div.__placeDot = null;
  const active = activeTid ?? [...pinned.keys()][0];
  if (active === undefined) return;
  const W = div.clientWidth || 400, H = div.clientHeight || 170;
  const m = { t: 16, r: 10, b: 16, l: 36 }, w = W - m.l - m.r, h = H - m.t - m.b;
  if (w < 30 || h < 30) return;
  const VMAX = D.vocabSize || 50000;
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([Math.log(VMAX), Math.log(1)]);
  const ctx = crispCanvas(d3.select(div), W, H);
  const cw = w / D.T, ch = h / D.L;
  for (let t = 0; t < D.T; t++) for (let l = 0; l < D.L; l++) {
    const r = rankAt(t, l, active);
    ctx.fillStyle = r >= 0 ? colorScale(Math.log(r + 1)) : '#f5f5f5';
    ctx.fillRect(m.l + t * cw, m.t + (D.L - 1 - l) * ch, cw + 0.5, ch + 0.5);
  }
  const svg = d3.select(div).append('svg').attr('width', W).attr('height', H).style('position', 'absolute').style('top', 0).style('left', 0);
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
  const x = d3.scaleLinear().domain([0, D.T - 1]).range([0, w]);
  const y = d3.scalePoint().domain(D.layers).range([h - ch / 2, ch / 2]);
  const yTicks = D.layers.filter((_, i) => i % Math.max(1, Math.round(D.L / 4)) === 0 || i === D.L - 1);
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(3));
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickValues(yTicks).tickSize(0).tickPadding(3));
  g.append('text').attr('class', 'chart-note').attr('x', w - 4).attr('y', h - 5).attr('text-anchor', 'end').attr('fill', '#999').attr('font-size', 9).text('Pos →');
  g.append('text').attr('class', 'chart-note').attr('transform', 'rotate(-90)').attr('x', -4).attr('y', 10).attr('text-anchor', 'end').attr('fill', '#999').attr('font-size', 9).text('Layer →');
  const lg = svg.append('g').attr('transform', `translate(${W - 110},2)`);
  for (let i = 0; i < 40; i++) lg.append('rect').attr('x', 18 + i * 1.5).attr('width', 1.5).attr('height', 8)
    .attr('fill', colorScale(Math.log(VMAX) * i / 39));
  lg.append('text').attr('x', 16).attr('y', 7).attr('text-anchor', 'end').attr('font-size', 8).attr('fill', '#666').text('rank 1');
  lg.append('text').attr('x', 80).attr('y', 7).attr('font-size', 8).attr('fill', '#666').text(d3.format('~s')(VMAX));
  const sel = g.append('circle').attr('r', 3).attr('fill', 'none').attr('stroke', 'var(--hl)').attr('stroke-width', 1.5).attr('pointer-events', 'none');
  const placeDot = () => sel.attr('cx', selCtx * cw + cw / 2).attr('cy', (D.L - 1 - layerIdx()) * ch + ch / 2);
  placeDot(); div.__placeDot = placeDot;
  g.append('rect').attr('width', w).attr('height', h).attr('fill', 'transparent').style('cursor', 'pointer')
    .on('mousemove click', e => {
      if (e.type === 'mousemove' && !e.shiftKey) return;
      const [mx, my] = d3.pointer(e);
      selCtx = Math.max(0, Math.min(D.T - 1, Math.floor(mx / cw)));
      selLayer = D.layers[Math.max(0, Math.min(D.L - 1, D.L - 1 - Math.floor(my / ch)))];
      render(); refreshCellPanel();
    });
}

// ---------------------------------------------------------------------------
// render loop (rAF-coalesced, incremental)
// ---------------------------------------------------------------------------

const lastKey = { hm: null, pinRow: null, byLayerP: null, byCtxP: null, byLayerC: null, byCtxC: null, rankHm: null };
let prevSelTok = -1, prevByLayerActive = -1, prevByCtxActive = -1, rafPending = false;

function schedule() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; render(); });
}

function render() {
  if (!D) return;
  const li = layerIdx();
  const sKey = [...pinned.entries()].map(([t, c]) => t + c).join(',') + '|' + showWs + '|' + D.ctxId;

  if (sKey !== lastKey.hm) { hmRepaint(true); lastKey.hm = sKey; }
  hmPlaceSel();

  if (prevSelTok !== selCtx || tokSpans.__ctx !== D.ctxId) {
    tokSpans.forEach((sp, i) => sp.classList.toggle('selected', i === selCtx));
    if (tokSpans[selCtx]) tokSpans[selCtx].scrollIntoView({ block: 'nearest' });
    prevSelTok = selCtx;
    tokSpans.__ctx = D.ctxId;
  }

  const pinRowKey = sKey + '|' + (activeTid ?? '');
  if (pinRowKey !== lastKey.pinRow) {
    const activeHm = activeTid ?? [...pinned.keys()][0];
    $('pinned').innerHTML = pinned.size === 0 ? '' :
      `<span class="pinned-label">pinned:</span>` + [...pinned].map(([tid, c]) =>
        `<button class="pinned-chip${tid === activeHm ? ' chip-active' : ''}" style="border-color:${c}${tid === activeHm ? `;background:${c}26` : ''}" data-tid="${tid}">${esc(clean(tid))}<span class="chip-x" data-tid="${tid}" title="unpin">×</span></button>`).join('')
      + `<button class="unpin-all" title="unpin all">× all</button>`;
    document.querySelectorAll('.pinned-chip').forEach(b => { b.onclick = () => { activeTid = +b.dataset.tid; render(); }; });
    document.querySelectorAll('.chip-x').forEach(x => { x.onclick = e => { e.stopPropagation(); pinToken(+x.dataset.tid); }; });
    const ua = document.querySelector('.unpin-all');
    if (ua) ua.onclick = () => { pinned.clear(); activeTid = null; refillPanels(); render(); };
    lastKey.pinRow = pinRowKey;
  }

  $('ctxLabel').textContent = selCtx;
  $('ctxTok').textContent = (D.pieces[selCtx] || '').replace(/\n/g, '⏎');
  $('layerLabel').textContent = selLayer;
  $('byLayerH').textContent = `By Layer (Pos ${selCtx} ${(D.pieces[selCtx] || '').replace(/\n/g, '⏎')})`;
  $('byCtxH').textContent = `By Pos (Layer ${selLayer})`;

  const byLayerPKey = selCtx + '|' + sKey;
  if (byLayerPKey !== lastKey.byLayerP) {
    for (const row of byLayerRows) fillRow(row.cells, selCtx, row.li);
    lastKey.byLayerP = byLayerPKey;
  }
  const byCtxPKey = selLayer + '|' + sKey;
  if (byCtxPKey !== lastKey.byCtxP) { byCtxRefill(li, true); lastKey.byCtxP = byCtxPKey; }
  if (prevByLayerActive !== li || lastKey.byLayerA !== D.ctxId) {
    byLayerRows.forEach(row => row.r.classList.toggle('active-row', row.li === li));
    prevByLayerActive = li; lastKey.byLayerA = D.ctxId;
  }
  if (prevByCtxActive !== selCtx) {
    byCtxScrollTo(selCtx);
    for (const row of byCtxRows) row.r.classList.toggle('active-row', +row.r.dataset.t === selCtx);
    prevByCtxActive = selCtx;
  }

  const chartBase = sKey + '|' + (activeTid ?? '') + '|' + rankCache.size + '|' + geomV;
  const byLayerCKey = selCtx + '|' + chartBase;
  if (byLayerCKey !== lastKey.byLayerC) {
    drawRankChart($('byLayerChart'), D.layers, 'Layer', (lidx, tid) => rankAt(selCtx, lidx, tid), v => { selLayer = v; });
    lastKey.byLayerC = byLayerCKey;
  }
  const byCtxCKey = selLayer + '|' + chartBase;
  if (byCtxCKey !== lastKey.byCtxC) {
    drawRankChart($('byCtxChart'), Array.from({length: D.T}, (_, i) => i), 'Pos', (t, tid) => rankAt(t, li, tid), v => { selCtx = v; });
    lastKey.byCtxC = byCtxCKey;
  }
  $('byLayerChart').__updateScrub?.(selLayer);
  $('byCtxChart').__updateScrub?.(selCtx);

  const rankHmKey = (activeTid ?? [...pinned.keys()][0] ?? '') + '|' + chartBase;
  if (rankHmKey !== lastKey.rankHm) {
    drawRankHm(); lastKey.rankHm = rankHmKey;
    const act = activeTid ?? [...pinned.keys()][0];
    $('rankTitle').innerHTML = act === undefined || !pinned.size ? '' :
      `rank of <b style="color:${pinned.get(act) || '#333'}">${esc(clean(act))}</b> at every (pos, layer) — click a pin to switch`;
  }
  $('rankHm').__placeDot?.();
}

// ---------------------------------------------------------------------------
// keyboard + resizing
// ---------------------------------------------------------------------------

function wireKeyboardAndResize() {
  document.addEventListener('keydown', e => {
    if (!D || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); selCtx = Math.max(0, selCtx - 1); render(); hmScrollTo(selCtx); refreshCellPanel(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); selCtx = Math.min(D.T - 1, selCtx + 1); render(); hmScrollTo(selCtx); refreshCellPanel(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); const i = layerIdx(); if (i < D.L - 1) { selLayer = D.layers[i + 1]; render(); refreshCellPanel(); } }
    if (e.key === 'ArrowDown') { e.preventDefault(); const i = layerIdx(); if (i > 0) { selLayer = D.layers[i - 1]; render(); refreshCellPanel(); } }
  });
  const forceRedraw = () => { geomV++; if (D) { hmSizeCanvas(); hmRepaint(true); render(); } };
  $('tokResize').onmousedown = e => {
    e.preventDefault();
    const startY = e.clientY, startH = toksDiv.offsetHeight;
    const onMove = e2 => { toksDiv.style.height = Math.max(20, startH - (e2.clientY - startY)) + 'px'; };
    const onUp = () => { removeEventListener('mousemove', onMove); removeEventListener('mouseup', onUp); forceRedraw(); };
    addEventListener('mousemove', onMove); addEventListener('mouseup', onUp);
  };
  document.querySelectorAll('.resize-handle').forEach(h => {
    h.onmousedown = e => {
      e.preventDefault();
      const col = +h.dataset.col, leftEl = $('col' + col);
      const startX = e.clientX, leftW = leftEl.offsetWidth;
      const onMove = e2 => { leftEl.style.width = Math.max(150, leftW + e2.clientX - startX) + 'px'; leftEl.style.flex = 'none'; };
      const onUp = () => { removeEventListener('mousemove', onMove); removeEventListener('mouseup', onUp); forceRedraw(); };
      addEventListener('mousemove', onMove); addEventListener('mouseup', onUp);
    };
  });
  addEventListener('resize', forceRedraw);
}

// ---------------------------------------------------------------------------
// tools column: cell readout + decomposition + generation
// ---------------------------------------------------------------------------

let cellFetchTimer = null;
function refreshCellPanel() {
  if (!D) return;
  $('cellLoc').textContent = `pos ${selCtx} · L${selLayer}`;
  $('decompOut').innerHTML = '';
  clearTimeout(cellFetchTimer);
  cellFetchTimer = setTimeout(async () => {
    try {
      const res = await api('/api/readout', { ctx_id: D.ctxId, pos: selCtx, layer: selLayer, top_n: 40 });
      if (res.pos !== selCtx || res.layer !== selLayer) return;
      const maxP = Math.max(...res.tokens.map(t => t.prob), 1e-9);
      $('cellReadout').innerHTML = res.tokens.map((t, i) => `
        <div class="cr-row" data-tid="${t.token}" title="token ${t.token} · logit ${t.logit.toFixed(2)}">
          <span class="cr-rank">${i}</span>
          <span class="cr-piece">${esc(showWs ? t.piece.replace(/ /g,'␣') : t.piece)}</span>
          <span class="cr-bar"><i style="width:${Math.round(100 * t.prob / maxP)}%"></i></span>
          <span class="cr-prob">${(t.prob * 100).toFixed(1)}%</span>
        </div>`).join('');
      $('cellReadout').querySelectorAll('.cr-row').forEach(row => {
        row.onclick = () => pinToken(+row.dataset.tid);
      });
    } catch (e) {
      $('cellReadout').innerHTML = `<span class="muted">${esc(e.message)}</span>`;
    }
  }, 120);
}

async function runDecompose() {
  if (!D) return;
  $('decompOut').innerHTML = '<span class="muted">decomposing…</span>';
  try {
    const res = await api('/api/decompose', { ctx_id: D.ctxId, pos: selCtx, layer: selLayer, k: 10 });
    const maxC = Math.max(...res.items.map(i => Math.abs(i.coeff)), 1e-9);
    $('decompOut').innerHTML =
      `<div class="muted" style="margin-top:6px">J-space decomposition (greedy pursuit):</div>` +
      res.items.map(it => `
        <div class="cr-row decomp-row" data-tid="${it.token}" title="coeff ${it.coeff.toFixed(3)} · cumulative variance explained ${(it.explained*100).toFixed(0)}%">
          <span class="cr-piece">${esc(it.piece)}</span>
          <span class="cr-bar"><i style="width:${Math.round(100 * Math.abs(it.coeff) / maxC)}%"></i></span>
          <span class="cr-prob">${it.coeff.toFixed(2)}</span>
        </div>`).join('');
    $('decompOut').querySelectorAll('.cr-row').forEach(row => { row.onclick = () => pinToken(+row.dataset.tid); });
  } catch (e) {
    $('decompOut').innerHTML = `<span class="muted">${esc(e.message)}</span>`;
  }
}

async function runGenerate() {
  if (!D) { setStatus('run a slice first', true); return; }
  $('genBtn').disabled = true;
  $('genOut').innerHTML = '<span class="muted">generating…</span>';
  try {
    const res = await api('/api/generate', {
      ctx_id: D.ctxId,
      n_predict: +$('genN').value || 24,
      sampling: $('genGreedy').checked ? { greedy: true } : { greedy: false, temp: +$('genTemp').value || 0.8 },
      interventions: activeSpecs(),
      compare: true,
    });
    let html = `<div class="gen-output"><span class="lbl">${activeSpecs().length ? 'steered' : 'output'}</span>${esc(res.steered.text) || '<i>(eos)</i>'}</div>`;
    if (res.baseline) html += `<div class="gen-output"><span class="lbl">baseline</span>${esc(res.baseline.text) || '<i>(eos)</i>'}</div>`;
    $('genOut').innerHTML = html;
  } catch (e) {
    $('genOut').innerHTML = `<span class="muted">${esc(e.message)}</span>`;
  } finally {
    $('genBtn').disabled = false;
  }
}

// ---------------------------------------------------------------------------
// interventions: chips + modal editor
// ---------------------------------------------------------------------------

function ivSummary(iv) {
  const lay = iv.layers ? (iv.layers[0] === iv.layers[1] ? `L${iv.layers[0]}` : `L${iv.layers[0]}–${iv.layers[1]}`) : 'all L';
  const pos = iv.pos[1] === -1 ? (iv.pos[0] === 0 ? 'all pos' : `pos ${iv.pos[0]}+`) : `pos ${iv.pos[0]}–${iv.pos[1] - 1}`;
  if (iv.type === 'steer') return `steer ${JSON.stringify(tokStr(iv.token))} α=${iv.alpha} ${lay} ${pos}`;
  if (iv.type === 'swap') return `swap ${JSON.stringify(tokStr(iv.tokenA))}↔${JSON.stringify(tokStr(iv.tokenB))} ${lay} ${pos}`;
  return `ablate ${JSON.stringify(tokStr(iv.token))} ${lay} ${pos}`;
}

function renderIvChips() {
  $('ivChips').innerHTML = interventions.map(iv => `
    <span class="ivchip ${iv.type}${iv.enabled ? '' : ' disabled'}" data-id="${iv.id}" title="click to toggle on/off">
      <span class="tag">${iv.type}</span> ${esc(ivSummary(iv).replace(iv.type + ' ', ''))}
      <button data-x="${iv.id}" title="remove">×</button>
    </span>`).join('');
  $('ivChips').querySelectorAll('.ivchip').forEach(chip => {
    chip.onclick = e => {
      if (e.target.dataset.x) return;
      const iv = interventions.find(v => v.id === +chip.dataset.id);
      iv.enabled = !iv.enabled;
      renderIvChips();
      if (D) runSlice();
    };
  });
  $('ivChips').querySelectorAll('button[data-x]').forEach(btn => {
    btn.onclick = () => {
      interventions = interventions.filter(v => v.id !== +btn.dataset.x);
      renderIvChips();
      if (D) runSlice();
    };
  });
}

function tokenSearchWidget(id, initial) {
  // returns {html, wire(onPick), get()}
  const state = { token: initial ?? null };
  const html = `
    <span class="tok-search" id="${id}">
      <input type="text" placeholder="search token… (or #id)" autocomplete="off"
             value="${state.token != null ? esc(tokStr(state.token)) : ''}">
      <div class="tok-results" style="display:none"></div>
    </span>`;
  const wire = () => {
    const root = $(id);
    const input = root.querySelector('input');
    const results = root.querySelector('.tok-results');
    let timer = null;
    const doSearch = async () => {
      const q = input.value;
      if (!q.trim()) { results.style.display = 'none'; return; }
      try {
        const res = await api('/api/search_tokens?q=' + encodeURIComponent(q) + '&limit=30');
        results.innerHTML = res.results.map(r =>
          `<div class="tok-result" data-tid="${r.token}"><span>${esc(JSON.stringify(r.piece))}</span><span class="tid">#${r.token}</span></div>`).join('')
          || '<div class="tok-result"><span class="muted">no match</span></div>';
        results.style.display = 'block';
        results.querySelectorAll('.tok-result[data-tid]').forEach(el => {
          el.onclick = () => {
            state.token = +el.dataset.tid;
            input.value = tokStr(state.token);
            results.style.display = 'none';
          };
        });
      } catch (e) { /* ignore */ }
    };
    input.oninput = () => { state.token = null; clearTimeout(timer); timer = setTimeout(doSearch, 150); };
    input.onfocus = () => { if (input.value && state.token == null) doSearch(); };
    input.onblur = () => setTimeout(() => { results.style.display = 'none'; }, 200);
  };
  return { html, wire, get: () => state.token };
}

let modalCtl = null;

function openIvModal(type, prefill = {}) {
  const fitted = PROPS.lens.source_layers;
  const lensDefault = D && fitted.includes(selLayer) ? selLayer : fitted[Math.floor(fitted.length / 2)] ?? 0;
  const posDefault = D ? selCtx : 0;
  const layerOpts = fitted.map(l => `<option value="${l}">${l}</option>`).join('');

  const w1 = tokenSearchWidget('tokSearch1', prefill.token ?? prefill.tokenA ?? null);
  const w2 = tokenSearchWidget('tokSearch2', prefill.tokenB ?? null);

  const titles = { steer: 'Steer — inject a J-lens direction', swap: 'Swap — patch one concept for another', ablate: 'Ablate — project a direction out' };
  $('ivTitle').textContent = titles[type];

  let rows = '';
  if (type === 'swap') {
    rows += `<div class="frow"><label>token A</label>${w1.html}</div>`;
    rows += `<div class="frow"><label>token B</label>${w2.html}</div>`;
  } else {
    rows += `<div class="frow"><label>token</label>${w1.html}</div>`;
  }
  if (type === 'steer') {
    rows += `<div class="frow"><label>strength α</label>
      <span class="alpha-row"><input type="range" id="ivAlpha" min="-8" max="8" step="0.25" value="${prefill.alpha ?? 3}">
      <span class="alpha-val" id="ivAlphaVal">${prefill.alpha ?? 3}</span></span></div>`;
  }
  const layerRow = type === 'swap'
    ? `<div class="frow"><label>layer</label>
         <select id="ivL0">${layerOpts}</select>
         <span class="fnote">swap re-reads coordinates per layer, so multi-layer swaps can cancel — one layer is the faithful patch</span></div>`
    : `<div class="frow"><label>layers</label>
         <span class="range-inputs"><select id="ivL0">${layerOpts}</select> to <select id="ivL1">${layerOpts}</select></span></div>`;
  rows += layerRow;
  rows += `<div class="frow"><label>positions</label>
    <span class="range-inputs">from <input type="number" id="ivP0" value="${posDefault}" min="0"> to
    <input type="number" id="ivP1" value="-1"> <span class="fnote">(-1 = end, incl. generated)</span></span></div>`;
  rows += `<div class="fnote">${{
    steer: 'Adds α · ‖h‖ · v̂ₜ to the residual stream at the chosen layers/positions. Positive α summons the concept; negative suppresses it.',
    swap: 'Reads the lens coordinates of tokens A and B out of the residual (c = V⁺h) and swaps them: h ← h + V(σ(c) − c). The component orthogonal to both directions is untouched.',
    ablate: 'Removes the projection of the residual onto the token\'s J-lens direction at every chosen layer (h ← h − v̂v̂ᵀh).',
  }[type]}</div>`;

  $('ivForm').innerHTML = rows;
  w1.wire(); if (type === 'swap') w2.wire();
  if (type === 'steer') {
    $('ivAlpha').oninput = () => { $('ivAlphaVal').textContent = $('ivAlpha').value; };
  }
  $('ivL0').value = String(lensDefault);
  if ($('ivL1')) $('ivL1').value = String(lensDefault);

  modalCtl = { type, w1, w2 };
  $('modalBack').style.display = 'flex';
  $('ivCancel').onclick = closeIvModal;
  $('modalBack').onclick = e => { if (e.target === $('modalBack')) closeIvModal(); };
  $('ivSave').onclick = saveIvModal;
}

function closeIvModal() { $('modalBack').style.display = 'none'; modalCtl = null; }

function saveIvModal() {
  const { type, w1, w2 } = modalCtl;
  const tok1 = w1.get(), tok2 = type === 'swap' ? w2.get() : null;
  if (tok1 == null || (type === 'swap' && tok2 == null)) {
    setStatus('pick a token from the search results first', true);
    return;
  }
  const l0 = +$('ivL0').value;
  const l1 = $('ivL1') ? +$('ivL1').value : l0;
  const p0 = Math.max(0, +$('ivP0').value || 0);
  let p1 = +$('ivP1').value;
  if (isNaN(p1) || p1 < 0) p1 = -1; else p1 = p1 + 1; // UI "to" is inclusive
  const iv = {
    id: ++ivCounter, type, enabled: true,
    layers: [Math.min(l0, l1), Math.max(l0, l1)],
    pos: [p0, p1],
  };
  if (type === 'steer') { iv.token = tok1; iv.alpha = +$('ivAlpha').value; }
  else if (type === 'ablate') { iv.token = tok1; }
  else { iv.tokenA = tok1; iv.tokenB = tok2; iv.layers = [l0, l0]; }
  interventions.push(iv);
  renderIvChips();
  closeIvModal();
  if (D || $('prompt').value.trim()) runSlice();
}
