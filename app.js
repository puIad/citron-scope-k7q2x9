import { classify, degrade, loadModel, normals } from './classify.js';

const NAMES = {
  barker: ['Barker code', 'Phase flips in a fixed binary rhythm'],
  cw: ['Plain tone', 'One steady frequency, no modulation'],
  fmcw: ['Linear chirp', 'Frequency sweeps in a straight line'],
  fsk: ['Frequency hops', 'Jumps between fixed frequencies'],
  hyp: ['Hyperbolic chirp', 'A sweep that follows a hyperbola'],
  nlfm: ['Non-linear chirp', 'A sweep that bends instead of running straight'],
  quad: ['Quadratic chirp', 'A sweep that follows a parabola'],
};
const SAMPLES = 33600, L = 16800;
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, parent) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
};

// ---------- formatting ----------
const dB = (v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)} dB`;
const minus = (s) => String(s).replace(/-/g, '−');
function describe(g) {
  const m = /^(-?\d+) MHz-(-?\d+) MHz$/.exec(g.frequency);
  const band = m ? `${minus(m[1] > 0 ? '+' + m[1] : m[1])} → ${minus(m[2] > 0 ? '+' + m[2] : m[2])} MHz` : `${minus(g.frequency)}`;
  const sweeps = !['barker', 'cw'].includes(g.label);
  return { duration: g.duration, bandwidth: sweeps ? g.bandwidth : null, band, carrier: !m };
}
function settingText(g) {
  const d = describe(g);
  return d.bandwidth ? `${d.duration} · ${d.bandwidth} wide · ${d.band}` : `${d.duration} · carrier ${d.band}`;
}
const pct = (p) => (p >= 0.9999 ? '>99.99' : p < 0.0001 ? '<0.01' : (100 * p).toFixed(p > 0.995 ? 2 : 1));
const ICON_OK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_NO = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/></svg>';

// ---------- canvas helpers ----------
function fit(canvas) {
  const r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return canvas.getContext('2d');
}

function drawTrace(canvas, x, ghost) {
  const ctx = fit(canvas), W = canvas.width, H = canvas.height, mid = H / 2;
  ctx.clearRect(0, 0, W, H);
  let peak = 1;
  for (let i = 0; i < SAMPLES; i++) peak = Math.max(peak, Math.abs(x[i]), ghost ? Math.abs(ghost[i]) : 0);
  const k = (H / 2 - 4) / peak, per = L / W;
  const band = (offset, color, alpha) => {
    ctx.fillStyle = color; ctx.globalAlpha = alpha;
    for (let c = 0; c < W; c++) {
      let lo = Infinity, hi = -Infinity;
      for (let n = Math.floor(c * per), e = Math.min(L, Math.floor((c + 1) * per) + 1); n < e; n++) {
        const v = x[2 * n + offset]; if (v < lo) lo = v; if (v > hi) hi = v;
      }
      ctx.fillRect(c, mid - hi * k, 1, Math.max(1, (hi - lo) * k));
    }
  };
  band(1, '#C9DD45', 0.55);
  band(0, '#E8D21D', 0.9);
  if (ghost) {
    ctx.globalAlpha = 0.85; ctx.strokeStyle = '#FBF8E8'; ctx.lineWidth = Math.max(1, W / 700);
    ctx.setLineDash([4 * W / 700 + 2, 4 * W / 700 + 2]);
    for (const sign of [1, -1]) {
      ctx.beginPath();
      for (let c = 0; c < W; c++) {
        let m = 0;
        for (let n = Math.floor(c * per), e = Math.min(L, Math.floor((c + 1) * per) + 1); n < e; n++) {
          m = Math.max(m, Math.hypot(ghost[2 * n], ghost[2 * n + 1]));
        }
        const y = mid - sign * m * k;
        c ? ctx.lineTo(c, y) : ctx.moveTo(c, y);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  ctx.globalAlpha = 1;
}

// Radix-2 FFT for the display-only spectrogram (the classifier never sees it).
const N = 256, HOP = 64, FRAMES = Math.floor((L - N) / HOP) + 1;
const hann = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
const rev = Uint16Array.from({ length: N }, (_, i) => { let r = 0; for (let b = 0, v = i; b < 8; b++, v >>= 1) r = (r << 1) | (v & 1); return r; });
const cosT = Float64Array.from({ length: N / 2 }, (_, i) => Math.cos((-2 * Math.PI * i) / N));
const sinT = Float64Array.from({ length: N / 2 }, (_, i) => Math.sin((-2 * Math.PI * i) / N));
function fft(re, im) {
  for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let size = 2; size <= N; size *= 2) {
    const half = size / 2, step = N / size;
    for (let s = 0; s < N; s += size) for (let j = 0; j < half; j++) {
      const a = s + j, b = a + half, wr = cosT[j * step], wi = sinT[j * step];
      const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
const STOPS = [[0, [34, 37, 14]], [0.3, [62, 85, 18]], [0.6, [168, 193, 42]], [0.82, [232, 210, 29]], [1, [251, 248, 232]]];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const t = i / 255; let k = 0; while (t > STOPS[k + 1][0]) k++;
  const [t0, c0] = STOPS[k], [t1, c1] = STOPS[k + 1], f = (t - t0) / (t1 - t0);
  return c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
});
const specImage = new OffscreenCanvas(FRAMES, N);
function drawSpectrogram(canvas, x) {
  const re = new Float64Array(N), im = new Float64Array(N), power = new Float32Array(FRAMES * N);
  for (let f = 0; f < FRAMES; f++) {
    for (let i = 0; i < N; i++) { const n = f * HOP + i; re[i] = x[2 * n] * hann[i]; im[i] = x[2 * n + 1] * hann[i]; }
    fft(re, im);
    for (let b = 0; b < N; b++) {
      const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-9);
      power[f * N + b] = v;
    }
  }
  // Colour climbs only above the noise floor (the median bin): pure static sits dim, energy shows bright.
  const sample = []; for (let i = 0; i < power.length; i += 11) sample.push(power[i]);
  sample.sort((a, b) => a - b);
  const floor = sample[sample.length >> 1] + 4, span = 34;
  const sctx = specImage.getContext('2d'), img = sctx.createImageData(FRAMES, N);
  for (let f = 0; f < FRAMES; f++) for (let b = 0; b < N; b++) {
    const shifted = (b + N / 2) % N, row = N - 1 - shifted;  // positive frequencies up
    const t = Math.max(0, Math.min(1, (power[f * N + b] - floor) / span));
    const c = LUT[Math.round(t * 255)], o = (row * FRAMES + f) * 4;
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const ctx = fit(canvas);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(specImage, 0, 0, canvas.width, canvas.height);
}

// ---------- scope ----------
const CX = 200, CY = 200, SEG = 360 / 7, GAP = 2.2;
const polar = (r, deg) => [CX + r * Math.cos((deg * Math.PI) / 180), CY + r * Math.sin((deg * Math.PI) / 180)];
const segStart = (c) => -90 - SEG / 2 + c * SEG;
function wedge(r0, r1, a0, a1) {
  const [x0, y0] = polar(r1, a0), [x1, y1] = polar(r1, a1), [x2, y2] = polar(r0, a1), [x3, y3] = polar(r0, a0);
  return `M${x3},${y3} L${x0},${y0} A${r1},${r1} 0 0 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 0 0 ${x3},${y3} Z`;
}
const scope = { juice: [], pct: [], labels: [], blips: [], best: null };
function buildRing(root, segs, labels, prefix, classes, wrap) {
  const defs = root.querySelector('defs') || svg('defs', {}, root);
  svg('path', { id: `${prefix}-top`, d: `M${CX},${CY + 177} a177,177 0 1,1 0,-354 a177,177 0 1,1 0,354` }, defs);  // from the bottom, clockwise
  svg('path', { id: `${prefix}-bottom`, d: `M${CX},${CY - 186} a186,186 0 1,0 0,372 a186,186 0 1,0 0,-372` }, defs);  // from the top, anticlockwise
  return classes.map((label, c) => {
    const a0 = segStart(c) + GAP, a1 = segStart(c + 1) - GAP, mid = (a0 + a1) / 2;
    const host = wrap ? wrap(c, label) : segs;
    const base = svg('path', { d: wedge(16, 164, a0, a1), class: 'seg-base' }, host);
    const juice = svg('path', { d: wedge(16, 164, a0, a1), class: 'seg-juice', opacity: 0 }, host);
    const deg = ((mid % 360) + 360) % 360, bottom = deg > 0 && deg < 180;
    const frac = bottom ? ((270 - deg + 360) % 360) / 360 : ((deg - 90 + 360) % 360) / 360;
    const text = svg('text', {}, labels);
    const tp = svg('textPath', { href: `#${prefix}-${bottom ? 'bottom' : 'top'}`, startOffset: `${(frac * 100).toFixed(2)}%`, 'text-anchor': 'middle' }, text);
    tp.textContent = NAMES[label][0];
    return { base, juice, label: text, mid, host };
  });
}
function buildScope(meta) {
  const segs = $('segments');
  buildRing($('scope'), segs, $('rind-labels'), 'rind', meta.classes).forEach((r) => {
    scope.juice.push(r.juice); scope.labels.push(r.label);
    const [tx, ty] = polar(116, r.mid);
    scope.pct.push(svg('text', { x: tx, y: ty, class: 'seg-pct' }, segs));
  });
  // the 660 templates as blips inside their modulation's segment
  const blips = $('blips'), seen = new Array(7).fill(0);
  const counts = meta.classes.map((l) => meta.configs.filter((c) => c.label === l).length);
  meta.configs.forEach((cfg) => {
    const c = meta.classes.indexOf(cfg.label), i = seen[c]++, n = counts[c];
    const a = segStart(c) + GAP + 3 + ((i * 0.6180339887) % 1) * (SEG - 2 * GAP - 6);
    const r = 34 + 118 * Math.sqrt((i + 0.5) / n);
    const [x, y] = polar(r, a);
    scope.blips.push(svg('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: 1.9, fill: '#FBF8E8', opacity: 0.12, class: 'blip' }, blips));
  });
  scope.ring = svg('circle', { r: 7, fill: 'none', stroke: '#C9DD45', 'stroke-width': 2, opacity: 0 }, blips);
  // sweep afterglow: thin wedges fading behind the arm
  const trail = $('sweep-trail'); trail.remove();
  const g = $('sweep');
  for (let s = 0; s < 14; s++) {  // nested wedges: overlap builds the fade, so there are no seams
    g.insertBefore(svg('path', { d: wedge(0, 168, -(s + 1) * 4, 0), fill: '#C9DD45', opacity: 0.032 }), g.firstChild);
  }
}
function paintScope(out) {
  out.probs.forEach((p, c) => {
    const win = c === out.label;
    scope.juice[c].setAttribute('opacity', win ? 1 : Math.min(0.6, Math.sqrt(p)).toFixed(3));
    scope.juice[c].classList.toggle('seg-win', win);
    scope.pct[c].textContent = p >= 0.0005 ? `${pct(p)}%` : '';
    scope.pct[c].classList.toggle('on', win);
    scope.labels[c].classList.toggle('on', win);
  });
  const sb = out.scores[out.best];
  out.scores.forEach((s, k) => {
    const v = Math.max(0, s / sb);
    scope.blips[k].setAttribute('opacity', (0.1 + 0.9 * v ** 4).toFixed(3));
    scope.blips[k].setAttribute('r', k === out.best ? 4 : 1.9);
  });
  const b = scope.blips[out.best];
  scope.ring.setAttribute('cx', b.getAttribute('cx')); scope.ring.setAttribute('cy', b.getAttribute('cy'));
  scope.ring.setAttribute('opacity', 1);
}

// ---------- validation curve under the slider ----------
function drawCurve(meta, snr) {
  const canvas = $('curve'), ctx = fit(canvas), W = canvas.width, H = canvas.height, d = W / canvas.clientWidth;
  const inset = 15 * d, xOf = (v) => inset + ((30 - v) / 60) * (W - 2 * inset);
  const top = 38 * d, bottom = H - 6 * d, yOf = (a) => top + ((1 - a) / 0.5) * (bottom - top);  // 50% to 100%
  ctx.clearRect(0, 0, W, H);
  ctx.font = `${11 * d}px JBMono, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  for (let v = 30; v >= -30; v -= 5) {
    const x = xOf(v), major = v % 10 === 0;
    ctx.fillStyle = major ? '#B9B48E' : '#6E7150';
    ctx.fillRect(x - 0.5 * d, 0, d, (major ? 7 : 4) * d);
    if (major) ctx.fillText(minus(v > 0 ? `+${v}` : `${v}`), x, 20 * d);
  }
  const pts = meta.validation.per_snr.map((p) => [xOf(p.snr), yOf(p.balanced_accuracy), p]);
  ctx.setLineDash([3 * d, 4 * d]); ctx.strokeStyle = '#4A4F24'; ctx.lineWidth = d;
  ctx.beginPath(); ctx.moveTo(inset, yOf(1)); ctx.lineTo(W - inset, yOf(1)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#C9DD45'; ctx.lineWidth = 2 * d; ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
  ctx.fillStyle = '#C9DD45'; pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 2.6 * d, 0, 7); ctx.fill(); });
  ctx.fillStyle = '#B9B48E'; ctx.textAlign = 'left'; ctx.fillText('100%', inset + 6 * d, yOf(1) + 15 * d);
  const last = pts[pts.length - 1];
  ctx.textAlign = 'right'; ctx.fillText(`${(100 * last[2].balanced_accuracy).toFixed(0)}%`, last[0], last[1] + 17 * d);
  const x = xOf(snr), y = yOf(curveAt(meta, snr).acc);
  ctx.fillStyle = '#E8D21D'; ctx.fillRect(x - d, top - 6 * d, 2 * d, bottom - top + 6 * d);
  ctx.beginPath(); ctx.arc(x, y, 4.5 * d, 0, 7); ctx.fill();
}
function curveAt(meta, snr) {
  const p = meta.validation.per_snr.slice().sort((a, b) => a.snr - b.snr);
  for (let i = 0; i < p.length - 1; i++) {
    if (snr >= p[i].snr && snr <= p[i + 1].snr) {
      const f = (snr - p[i].snr) / (p[i + 1].snr - p[i].snr);
      return { acc: p[i].balanced_accuracy + f * (p[i + 1].balanced_accuracy - p[i].balanced_accuracy), exact: f === 0 || f === 1 };
    }
  }
  return { acc: p[p.length - 1].balanced_accuracy, exact: true };
}

// ---------- state ----------
const state = { model: null, meta: null, gallery: null, pick: 17, snr: 30, seed: 1, normal: null, lastLabel: -1, frame: 0 };
const pulse = (i) => state.gallery.subarray(i * SAMPLES, (i + 1) * SAMPLES);

function render() {
  state.frame = 0;
  const { meta } = state, g = meta.gallery[state.pick], clean = pulse(state.pick);
  const x = degrade(clean, state.snr, state.normal);
  const out = classify(state.model, x);
  const cls = meta.classes[out.label], truth = meta.classes.indexOf(g.label);
  const matched = meta.configs[out.best];
  $('verdict-class').textContent = NAMES[cls][0];
  const right = out.label === truth, exact = matched.name === g.name;
  $('verdict-line').innerHTML =
    `<span class="mono code">${cls}</span> <span class="mono">${pct(out.probs[out.label])}%</span> sure. ` +
    `Best match: <span class="mono">${settingText(matched)}</span>. ` +
    (right
      ? `<span class="ok">${ICON_OK}Right${exact ? ', and the exact setting it was recorded with.' : '.'}</span>`
      : `<span class="no">${ICON_NO}Wrong: it was ${NAMES[g.label][0].toLowerCase()}.</span>`);
  paintScope(out);
  drawTrace($('trace'), x, clean);
  drawSpectrogram($('spectrogram'), x);
  $('snr-out').textContent = dB(state.snr);
  $('snr-input').setAttribute('aria-valuetext', dB(state.snr));
  drawCurve(meta, state.snr);
  const c = curveAt(meta, state.snr);
  $('curve-note').innerHTML = c.exact
    ? `At <b>${dB(state.snr)}</b>: <b>${(100 * c.acc).toFixed(1)}%</b> of 13,200 validation pulses named correctly (balanced).`
    : `At <b>${dB(state.snr)}</b>: about <b>${(100 * c.acc).toFixed(0)}%</b> (measured every 5 dB).`;
  if (out.label !== state.lastLabel) { $('live').textContent = `The scope says ${NAMES[cls][0]}.`; state.lastLabel = out.label; }
}
const schedule = () => { if (!state.frame) state.frame = requestAnimationFrame(render); };

// ---------- gallery ----------
function buildShelf() {
  const { meta } = state, shelf = $('shelf');
  meta.classes.forEach((label) => {
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = `<div class="row-head"><h3>${NAMES[label][0]}</h3><p>${NAMES[label][1]}. <span class="mono">${label}</span></p></div><div class="tiles"></div>`;
    const tiles = row.querySelector('.tiles');
    meta.gallery.forEach((g, i) => {
      if (g.label !== label) return;
      const d = describe(g), b = document.createElement('button');
      b.type = 'button'; b.className = 'tile'; b.dataset.index = i;
      b.setAttribute('aria-pressed', i === state.pick);
      b.setAttribute('aria-label', `${NAMES[label][0]}, ${settingText(g)}`);
      b.innerHTML = `<canvas aria-hidden="true"></canvas><span class="lab"><span>len</span><b>${d.duration}</b>` +
        (d.bandwidth ? `<span>bw</span><b>${d.bandwidth}</b><span>band</span><b>${d.band}</b>` : `<span>tone</span><b>${d.band}</b>`) + '</span>';
      tiles.appendChild(b);
    });
    shelf.appendChild(row);
  });
  shelf.addEventListener('click', (e) => {
    const b = e.target.closest('.tile'); if (!b) return;
    choose(+b.dataset.index);
    $('main').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' });
  });
  requestAnimationFrame(paintShelf);
}
function paintShelf() {
  document.querySelectorAll('.tile').forEach((b) => drawSpectrogram(b.querySelector('canvas'), pulse(+b.dataset.index)));
}
function choose(i) {
  state.pick = i; state.lastLabel = -1;
  document.querySelectorAll('.tile').forEach((b) => b.setAttribute('aria-pressed', +b.dataset.index === i));
  render();
}

// ---------- game ----------
const game = { level: -10, round: 0, you: 0, scope: 0, current: null, open: false, ring: [] };
function exemplar(label) { return state.meta.gallery.findIndex((g) => g.label === label && g.duration.startsWith('100')); }
function buildGame() {
  const { classes } = state.meta, root = $('answer-scope');
  game.ring = buildRing(root, $('ans-segments'), $('ans-labels'), 'ans', classes, (c, label) => {
    const g = svg('g', { class: 'ans', role: 'button', tabindex: 0, 'aria-label': NAMES[label][0], 'data-class': c }, $('ans-segments'));
    g.addEventListener('click', () => answer(c));
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); answer(c); } });
    return g;
  });
  const ex = $('exemplars');
  classes.forEach((label) => {
    const item = document.createElement('div'); item.className = 'ex';
    item.innerHTML = `<canvas aria-hidden="true"></canvas><span>${NAMES[label][0]}</span>`;
    ex.appendChild(item);
  });
  requestAnimationFrame(() => [...ex.children].forEach((item, c) => drawSpectrogram(item.querySelector('canvas'), pulse(exemplar(classes[c])))));
  document.querySelectorAll('.levels button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.levels button').forEach((o) => o.setAttribute('aria-checked', o === b));
    game.level = +b.dataset.snr; newGame();
  }));
  newGame();
}
function newGame() { Object.assign(game, { round: 0, you: 0, scope: 0 }); nextRound(); }
function nextRound() {
  game.round++; game.open = true;
  const pick = Math.floor(Math.random() * state.meta.gallery.length);
  const x = degrade(pulse(pick), game.level, normals(SAMPLES, (Math.random() * 2 ** 32) >>> 0));
  game.current = { pick, x, truth: state.meta.classes.indexOf(state.meta.gallery[pick].label) };
  drawSpectrogram($('arena-spec'), x);
  drawTrace($('arena-trace'), x, null);
  $('arena-snr').textContent = dB(game.level);
  $('answer-scope').querySelectorAll('.ans-mark, .ans-miss').forEach((m) => m.remove());
  game.ring.forEach((r) => {
    r.juice.setAttribute('opacity', 0); r.host.classList.remove('locked');
    r.host.setAttribute('aria-disabled', 'false'); r.label.classList.remove('on');
  });
  $('reveal').innerHTML = '<p>Which modulation is buried in there? Compare it with the clean examples below.</p>';
  $('answer-hint').hidden = false;
  tally();
}
function mark(c, text, cls, onJuice, r) {
  const [x, y] = polar(r, game.ring[c].mid);
  const t = svg('text', { x, y, class: `ans-mark ${cls}${onJuice ? ' on-juice' : ''}` }, game.ring[c].host);
  t.textContent = text;
}
function answer(guess) {
  if (!game.open) return;
  game.open = false;
  $('answer-hint').hidden = true;
  const { x, truth, pick } = game.current, out = classify(state.model, x), names = state.meta.classes;
  game.you += guess === truth; game.scope += out.label === truth;
  game.ring.forEach((r) => { r.host.classList.add('locked'); r.host.setAttribute('aria-disabled', 'true'); });
  game.ring[truth].juice.setAttribute('opacity', 1); game.ring[truth].label.classList.add('on');
  if (guess !== truth) {
    const a0 = segStart(guess) + GAP + 1.2, a1 = segStart(guess + 1) - GAP - 1.2;
    svg('path', { d: wedge(18, 162, a0, a1), class: 'ans-miss' }, game.ring[guess].host);
  }
  const same = out.label === guess;
  mark(guess, 'YOU', 'you', guess === truth, same ? 122 : 100);
  mark(out.label, 'SCOPE', 'scope-m', out.label === truth, same ? 82 : 100);
  const g = state.meta.gallery[pick], last = game.round >= 10;
  const say = guess === truth && out.label === truth ? 'You both got it.'
    : guess === truth ? 'You got it and the scope did not. Rare: screenshot this.'
      : out.label === truth ? `The scope got it (${pct(out.probs[out.label])}% sure); you said ${NAMES[names[guess]][0].toLowerCase()}.`
        : 'You both missed this one.';
  const end = last ? ` <b>Final: you ${game.you}, scope ${game.scope}.</b>` : '';
  $('reveal').innerHTML = `<p>It was <b>${NAMES[g.label][0].toLowerCase()}</b>, ${settingText(g)}. ${say}${end}</p>` +
    `<button type="button" class="cta" id="next">${last ? 'Play again' : 'Next pulse'}</button>`;
  $('next').addEventListener('click', () => (last ? newGame() : nextRound()));
  $('next').focus({ preventScroll: true });
  tally();
}
function tally() { $('tally').textContent = `You ${game.you} · Scope ${game.scope} · round ${Math.min(game.round, 10)} of 10`; }

// ---------- loading ----------
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!res.body) return res.arrayBuffer();
  // Content-Length is the compressed size when the host gzips (GitHub Pages does), so it is only a total when uncompressed.
  const total = res.headers.get('content-encoding') ? 0 : +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.length; }
  return buf.buffer;
}
function introGlide() {
  if (reduceMotion) return;
  const t0 = performance.now(), from = 30, to = -18, ms = 3200;
  let stop = false;
  const cancel = () => { stop = true; };
  $('snr-input').addEventListener('pointerdown', cancel, { once: true });
  $('snr-input').addEventListener('keydown', cancel, { once: true });
  const step = (t) => {
    if (stop) return;
    const f = Math.min(1, (t - t0) / ms), e = f < 0.5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2;
    const v = Math.round(from + (to - from) * e);
    if (v !== state.snr) { state.snr = v; $('snr-input').value = v; render(); }
    if (f < 1) requestAnimationFrame(step);
  };
  setTimeout(() => requestAnimationFrame(step), 600);
}

async function main() {
  document.body.classList.add('loading');
  const meta = await (await fetch('data/meta.json')).json();
  state.meta = meta;
  buildScope(meta);
  const [gallery, templates] = await Promise.all([
    fetchBytes('data/gallery.bin', () => {}),
    fetchBytes('data/templates.bin', (got, total) => {
      $('verdict-line').innerHTML = `Loading the 660 templates: <span class="mono">${(got / 1e6).toFixed(1)}${total ? ` / ${(total / 1e6).toFixed(1)}` : ''} MB</span>`;
    }),
  ]);
  state.gallery = new Int16Array(gallery);
  state.model = loadModel(meta, templates);
  state.normal = normals(SAMPLES, state.seed);
  document.body.classList.remove('loading');
  render();
  buildShelf();
  buildGame();
  $('snr-input').addEventListener('input', (e) => { state.snr = +e.target.value; schedule(); });
  $('redraw').addEventListener('click', () => { state.normal = normals(SAMPLES, ++state.seed); render(); });
  let resize;
  addEventListener('resize', () => { clearTimeout(resize); resize = setTimeout(() => { render(); paintShelf(); }, 150); });
  introGlide();
}
main().catch((err) => {
  $('verdict-class').textContent = 'Oops';
  $('verdict-line').textContent = `Couldn’t load the model (${err.message}). Serve this folder over http, not file://, and reload.`;
  console.error(err);
});
