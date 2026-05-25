// ================================================================
// ESN.js — Echo State Network (ORION Stage 1)
// Optimised for browser: N=40, typed Float64Arrays throughout
// ================================================================

// ---- Seeded RNG (reproducible, fast) ----
function seededRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (s >>> 0) / 4294967296 * 2 - 1;
  };
}

// ---- Build reservoir weight matrix with desired spectral radius ----
// Uses power-iteration to estimate dominant eigenvalue, then rescales
function buildReservoirMatrix(N, spectralRadius, seed) {
  const rng = seededRng(seed);
  const W = new Float64Array(N * N);

  // Sparse connectivity (~25% density)
  for (let i = 0; i < N * N; i++) {
    const r = rng();
    W[i] = Math.abs(r) > 0.75 ? rng() : 0;
  }

  // Power iteration (20 iters) to estimate spectral radius
  let v = new Float64Array(N);
  for (let i = 0; i < N; i++) v[i] = rng();
  let norm = 0;
  for (let i = 0; i < N; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < N; i++) v[i] /= norm;

  for (let iter = 0; iter < 20; iter++) {
    const Wv = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      const row = i * N;
      for (let j = 0; j < N; j++) s += W[row + j] * v[j];
      Wv[i] = s;
    }
    norm = 0;
    for (let i = 0; i < N; i++) norm += Wv[i] * Wv[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < N; i++) v[i] = Wv[i] / norm;
  }

  // Rayleigh quotient → approximate spectral radius
  let sr = 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    const row = i * N;
    for (let j = 0; j < N; j++) s += W[row + j] * v[j];
    sr += s * v[i];
  }
  sr = Math.abs(sr) || 1;

  const scale = spectralRadius / sr;
  for (let i = 0; i < N * N; i++) W[i] *= scale;
  return W;
}

// ---- Build input weight matrix ----
function buildInputMatrix(N, F, seed) {
  const rng = seededRng(seed);
  const W = new Float64Array(N * F);
  for (let i = 0; i < N * F; i++) W[i] = rng() * 0.5;
  return W;
}

// ---- Ridge regression via Cholesky on normal equations ----
// Solves (S'S + λI) w = S'y  →  uses Gaussian elimination on (N)×(N+1)
function ridgeRegression(S, Y, N, T, lambda) {
  // Build S'S (N×N) and S'y (N)
  const StS = new Float64Array(N * N);
  const Sty = new Float64Array(N);

  for (let t = 0; t < T; t++) {
    const st = S[t]; // length-N array
    for (let i = 0; i < N; i++) {
      Sty[i] += st[i] * Y[t];
      const rowI = i * N;
      for (let j = i; j < N; j++) {
        const val = st[i] * st[j];
        StS[rowI + j] += val;
        if (i !== j) StS[j * N + i] += val;
      }
    }
  }
  // Regularise diagonal
  for (let i = 0; i < N; i++) StS[i * N + i] += lambda;

  // Gaussian elimination with partial pivoting
  // Build augmented [StS | Sty] as plain arrays for speed
  const M = [];
  for (let i = 0; i < N; i++) {
    const row = new Float64Array(N + 1);
    const src = i * N;
    for (let j = 0; j < N; j++) row[j] = StS[src + j];
    row[N] = Sty[i];
    M.push(row);
  }

  for (let col = 0; col < N; col++) {
    // Find pivot
    let maxRow = col, maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < N; row++) {
      const v = Math.abs(M[row][col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) { const tmp = M[col]; M[col] = M[maxRow]; M[maxRow] = tmp; }

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-14) continue;

    for (let row = col + 1; row < N; row++) {
      const f = M[row][col] / pivot;
      if (f === 0) continue;
      for (let j = col; j <= N; j++) M[row][j] -= f * M[col][j];
    }
  }

  // Back substitution
  const x = new Float64Array(N);
  for (let i = N - 1; i >= 0; i--) {
    let s = M[i][N];
    for (let j = i + 1; j < N; j++) s -= M[i][j] * x[j];
    x[i] = s / (M[i][i] || 1e-14);
  }
  return x;
}

// ================================================================
// ORION Engine — N=40 neurons for smooth browser performance
// ================================================================
class ORIONEngine {
  constructor(cfg = {}) {
    this.N   = cfg.N   || 40;
    this.F   = cfg.F   || 6;
    this.sr  = cfg.sr  || 0.9;
    this.a   = cfg.a   || 0.3;      // leaking rate
    this.beta= cfg.beta|| 0.05;     // oscillation amplitude
    this.omega = 2 * Math.PI / 24;  // daily cycle
    this.phi   = 0;
    this.lambda= cfg.lambda || 1e-3;

    this.W_res = buildReservoirMatrix(this.N, this.sr, 42);
    this.W_in  = buildInputMatrix(this.N, this.F, 99);
    this.W_out = null;
    this.state = new Float64Array(this.N);
  }

  featureVector(row, ns) {
    const n = (v, k) => (ns[k].max - ns[k].min) > 0
      ? (v - ns[k].min) / (ns[k].max - ns[k].min) : 0;
    return [
      n(row.irr, 'irr'), n(row.tmp, 'tmp'),
      n(row.hum, 'hum'), n(row.wnd, 'wnd'),
      Math.sin(2 * Math.PI * row.h / 24),
      Math.cos(2 * Math.PI * row.h / 24),
    ];
  }

  _step(x, t) {
    const N = this.N, W = this.W_res, Wi = this.W_in, F = this.F;
    const osc = this.beta * Math.sin(this.omega * t + this.phi);
    const newS = new Float64Array(N);

    for (let i = 0; i < N; i++) {
      let s = osc;
      const rowW = i * N, rowI = i * F;
      for (let j = 0; j < N; j++) s += W[rowW + j] * this.state[j];
      for (let j = 0; j < F; j++) s += Wi[rowI + j] * x[j];
      newS[i] = (1 - this.a) * this.state[i] + this.a * Math.tanh(s);
    }
    this.state = newS;
    return newS;
  }

  collectStates(data, ns) {
    this.state = new Float64Array(this.N);
    return data.map((row, t) => this._step(this.featureVector(row, ns), t));
  }

  train(data, ns) {
    const states = this.collectStates(data, ns);
    const Y = data.map(d => d.pwr / (ns.pwr.max || 1));
    this.W_out = ridgeRegression(states, Y, this.N, data.length, this.lambda);
    return states;
  }

  predict(data, ns) {
    if (!this.W_out) return data.map(() => 0);
    const states = this.collectStates(data, ns);
    const Wo = this.W_out, mx = ns.pwr.max || 1;
    return states.map(s => {
      let v = 0;
      for (let j = 0; j < this.N; j++) v += s[j] * Wo[j];
      return Math.max(0, v * mx);
    });
  }

  predictOne(feat, ns, t = 12) {
    if (!this.W_out) return 0;
    const s = this._step(feat, t);
    let v = 0;
    for (let j = 0; j < this.N; j++) v += s[j] * this.W_out[j];
    return Math.max(0, v * (ns.pwr.max || 1));
  }

  getSnapshot() { return Array.from(this.state); }
}

// ---- Normalization stats ----
function computeNormStats(data) {
  const stats = {};
  for (const k of ['irr','tmp','hum','wnd','pwr']) {
    const vals = data.map(d => d[k]).filter(Number.isFinite);
    let mn = Infinity, mx = -Infinity, sm = 0;
    for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v; sm += v; }
    stats[k] = { min: mn, max: mx, mean: sm / (vals.length || 1) };
  }
  return stats;
}

window.ORIONEngine     = ORIONEngine;
window.computeNormStats = computeNormStats;
