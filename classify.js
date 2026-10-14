// The frozen a_templates bank, run in JS. Same math as browser_logits() in demo/export.py:
// s_k = |<x, t_k>|^2 / (|t_k|^2 * noise power), class logit = log mean exp(s_k) over the class's templates.
// Shared by the page and test_classify.mjs.

export function loadModel(meta, buffer) {
  const codes = new Int8Array(buffer);
  const configs = meta.configs;
  const energy = new Float64Array(configs.length);
  const classOf = new Int32Array(configs.length);
  configs.forEach((c, k) => {
    let e = 0;
    for (let i = c.offset, end = c.offset + 2 * c.length; i < end; i++) e += codes[i] * codes[i];
    energy[k] = e;
    classOf[k] = meta.classes.indexOf(c.label);
  });
  return { meta, codes, energy, classOf };
}

// make_dataset_hackathon.degrade(): noise power from 0..20 us, signal power from 21..119 us, complex
// Gaussian noise to reach the requested SNR, clip to int16, truncate toward zero. `normal` is a fixed
// standard-normal draw (length 33600), so moving the SNR only rescales the same noise.
export function degrade(clean, snrDb, normal, fs = 120e6) {
  const noiseEnd = Math.trunc(20e-6 * fs), sigStart = Math.trunc(21e-6 * fs), sigEnd = Math.trunc(119e-6 * fs);
  const power = (a, b) => {
    let s = 0;
    for (let n = a; n < b; n++) s += clean[2 * n] ** 2 + clean[2 * n + 1] ** 2;
    return s / (b - a);
  };
  const noise = power(0, noiseEnd);
  const added = (power(sigStart, sigEnd) - noise) / 10 ** (snrDb / 10) - noise;
  const sigma = Math.sqrt(Math.max(0, added) / 2);
  const out = new Int16Array(clean.length);
  for (let i = 0; i < clean.length; i++) {
    out[i] = Math.trunc(Math.min(32767, Math.max(-32768, clean[i] + sigma * normal[i])));
  }
  return out;
}

export function classify(model, x) {
  const { meta, codes, energy, classOf } = model;
  let noise = 0;
  for (let n = 0; n < meta.noise_end; n++) noise += x[2 * n] ** 2 + x[2 * n + 1] ** 2;
  noise = Math.max(noise / meta.noise_end, 1e-6);

  const K = meta.configs.length;
  const scores = new Float64Array(K);
  let best = 0;
  for (let k = 0; k < K; k++) {
    const c = meta.configs[k];
    let re = 0, im = 0;
    for (let i = 0, j = c.offset, p = 2 * c.start; i < c.length; i++, j += 2, p += 2) {
      const tr = codes[j], ti = codes[j + 1], xr = x[p], xi = x[p + 1];
      re += xr * tr + xi * ti;  // x * conj(t)
      im += xi * tr - xr * ti;
    }
    scores[k] = (re * re + im * im) / (energy[k] * noise);
    if (scores[k] > scores[best]) best = k;
  }

  const C = meta.classes.length;
  const logits = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    let m = -Infinity, count = 0;
    for (let k = 0; k < K; k++) if (classOf[k] === c && scores[k] > m) m = scores[k];
    let sum = 0;
    for (let k = 0; k < K; k++) if (classOf[k] === c) { sum += Math.exp(scores[k] - m); count++; }
    logits[c] = m + Math.log(sum) - Math.log(count);
  }
  const top = Math.max(...logits);
  const exp = Array.from(logits, (l) => Math.exp(l - top));
  const total = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((e) => e / total);
  return { logits, probs, scores, best, label: probs.indexOf(Math.max(...probs)) };
}

// Seeded standard normals (mulberry32 + Box-Muller) for the noise draw.
export function normals(n, seed) {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - rand())), th = 2 * Math.PI * rand();
    out[i] = r * Math.cos(th);
    if (i + 1 < n) out[i + 1] = r * Math.sin(th);
  }
  return out;
}
