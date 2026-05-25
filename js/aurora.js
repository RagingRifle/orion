// ================================================================
// aurora.js — AURORA Adaptive Refinement Stage (fast version)
// ================================================================

class AURORAEngine {
  constructor(cfg = {}) {
    this.windowSize       = cfg.windowSize       || 24;
    this.radiantScale     = cfg.radiantScaleFactor || 0.7;
    this.adaptiveGain     = 1.0;
    this.weights          = null; // 8-element array (tiny, fast to solve)
  }

  _feat(row, orionPred, ns) {
    const n = (v, k) => (ns[k].max - ns[k].min) > 0
      ? (v - ns[k].min) / (ns[k].max - ns[k].min) : 0;
    return [
      orionPred / (ns.pwr.max || 1),
      n(row.irr, 'irr'), n(row.tmp, 'tmp'),
      n(row.hum, 'hum'), n(row.wnd, 'wnd'),
      Math.sin(2 * Math.PI * row.h / 24),
      Math.cos(2 * Math.PI * row.h / 24),
      1.0, // bias
    ];
  }

  _variance(arr) {
    if (!arr.length) return 0;
    let sm = 0;
    for (const v of arr) sm += v;
    const mn = sm / arr.length;
    let v2 = 0;
    for (const v of arr) v2 += (v - mn) ** 2;
    return v2 / arr.length;
  }

  // Tiny 8×8 least-squares — runs in microseconds
  _lsq(X, Y) {
    const m = X[0].length, n = X.length;
    const XtX = [];
    for (let i = 0; i < m; i++) {
      XtX.push(new Float64Array(m));
      for (let j = 0; j < m; j++) {
        let s = i === j ? 1e-3 : 0; // regularise
        for (let t = 0; t < n; t++) s += X[t][i] * X[t][j];
        XtX[i][j] = s;
      }
    }
    const XtY = new Float64Array(m);
    for (let i = 0; i < m; i++)
      for (let t = 0; t < n; t++)
        XtY[i] += X[t][i] * Y[t];

    // Gaussian elimination (8×8 — trivial)
    const M = XtX.map((row, i) => { const r = Array.from(row); r.push(XtY[i]); return r; });
    for (let col = 0; col < m; col++) {
      let mx = col;
      for (let r = col + 1; r < m; r++) if (Math.abs(M[r][col]) > Math.abs(M[mx][col])) mx = r;
      [M[col], M[mx]] = [M[mx], M[col]];
      const p = M[col][col] || 1e-12;
      for (let r = col + 1; r < m; r++) {
        const f = M[r][col] / p;
        for (let j = col; j <= m; j++) M[r][j] -= f * M[col][j];
      }
    }
    const x = new Float64Array(m);
    for (let i = m - 1; i >= 0; i--) {
      let s = M[i][m];
      for (let j = i + 1; j < m; j++) s -= M[i][j] * x[j];
      x[i] = s / (M[i][i] || 1e-12);
    }
    return x;
  }

  train(data, orionPreds, ns) {
    const X = data.map((row, i) => this._feat(row, orionPreds[i], ns));
    const residuals = data.map((d, i) => d.pwr - orionPreds[i]);
    this.weights = this._lsq(X, residuals);

    const varRes   = this._variance(residuals);
    const varOrion = this._variance(orionPreds);
    this.adaptiveGain = Math.min(1.0, varRes / (varOrion + 1e-6)) * this.radiantScale;
  }

  predict(data, orionPreds, ns) {
    if (!this.weights) return [...orionPreds];
    const allPwr = data.map(d => d.pwr);
    const globalVar = this._variance(allPwr);

    return data.map((row, i) => {
      const x = this._feat(row, orionPreds[i], ns);
      let corr = 0;
      for (let j = 0; j < this.weights.length; j++) corr += x[j] * this.weights[j];
      corr *= ns.pwr.max || 1;

      // Radiant scaling: amplify correction when local error is high
      const wStart = Math.max(0, i - this.windowSize);
      const localPwr = allPwr.slice(wStart, i + 1);
      const localVar = this._variance(localPwr);
      const rScale   = Math.min(2, Math.sqrt(localVar / (globalVar + 1e-6)));
      return Math.max(0, orionPreds[i] + corr * rScale * this.adaptiveGain);
    });
  }

  predictOne(feat8, orionPred, ns) {
    if (!this.weights) return orionPred;
    let corr = 0;
    for (let j = 0; j < this.weights.length; j++) corr += feat8[j] * this.weights[j];
    corr *= (ns.pwr.max || 1) * this.adaptiveGain * 0.3;
    return Math.max(0, orionPred + corr);
  }
}

window.AURORAEngine = AURORAEngine;
