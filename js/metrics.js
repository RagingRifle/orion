// ================================================================
// metrics.js — Evaluation metrics + benchmark reference data
// ================================================================

function computeRMSE(actual, predicted) {
  const n = actual.length;
  const mse = actual.reduce((s, a, i) => s + (a - predicted[i]) ** 2, 0) / n;
  return Math.sqrt(mse);
}

function computeMAE(actual, predicted) {
  const n = actual.length;
  return actual.reduce((s, a, i) => s + Math.abs(a - predicted[i]), 0) / n;
}

function computeMAPE(actual, predicted) {
  let count = 0, sum = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] > 1e-3) { // Avoid division by near-zero
      sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
      count++;
    }
  }
  return count > 0 ? (sum / count) * 100 : 0;
}

function computeR2(actual, predicted) {
  const mean = actual.reduce((s, v) => s + v, 0) / actual.length;
  const ss_tot = actual.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ss_res = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  return 1 - ss_res / (ss_tot || 1);
}

function computeBaselinePredictions(data) {
  const mean = data.reduce((s, d) => s + d.pwr, 0) / data.length;
  return data.map(() => mean);
}

// ---- Paper benchmark reference values (from the paper) ----
// Normalized to kW scale relative to the UCI and AEP datasets
const PAPER_BENCHMARKS = {
  uci: [
    { model: 'VGG16',               rmse: 8.72,  mae: 5.61, mape: 4.82 },
    { model: 'ResNet50',            rmse: 7.94,  mae: 5.03, mape: 4.31 },
    { model: 'EfficientNet-B0',     rmse: 7.48,  mae: 4.77, mape: 3.97 },
    { model: 'Inception-v3',        rmse: 7.21,  mae: 4.55, mape: 3.76 },
    { model: 'DenseNet121',         rmse: 6.94,  mae: 4.32, mape: 3.54 },
    { model: 'CNN-LSTM Hybrid',     rmse: 6.42,  mae: 3.98, mape: 3.12 },
    { model: 'Attention-LSTM',      rmse: 5.87,  mae: 3.61, mape: 2.71 },
    { model: 'Transformer-TFT',     rmse: 4.93,  mae: 2.98, mape: 2.18 },
    { model: 'ORION only',          rmse: 4.12,  mae: 2.41, mape: 1.87, isOrion: true },
    { model: 'ORION + AURORA',      rmse: 3.50,  mae: 2.00, mape: 1.50, isAurora: true },
  ],
  aep: [
    { model: 'VGG16',               rmse: 9.88,  mae: 6.44, mape: 5.73 },
    { model: 'ResNet50',            rmse: 8.95,  mae: 5.82, mape: 5.14 },
    { model: 'EfficientNet-B0',     rmse: 8.43,  mae: 5.41, mape: 4.72 },
    { model: 'Inception-v3',        rmse: 8.12,  mae: 5.17, mape: 4.43 },
    { model: 'DenseNet121',         rmse: 7.81,  mae: 4.93, mape: 4.12 },
    { model: 'CNN-LSTM Hybrid',     rmse: 7.28,  mae: 4.56, mape: 3.68 },
    { model: 'Attention-LSTM',      rmse: 6.71,  mae: 4.14, mape: 3.22 },
    { model: 'Transformer-TFT',     rmse: 5.62,  mae: 3.43, mape: 2.64 },
    { model: 'ORION only',          rmse: 4.68,  mae: 2.88, mape: 2.21, isOrion: true },
    { model: 'ORION + AURORA',      rmse: 4.00,  mae: 2.50, mape: 2.00, isAurora: true },
  ]
};

// Scale metrics to match dataset magnitudes
// Paper reports % errors — we scale RMSE/MAE to kW units of each dataset
function scaleMetricsToDataset(metrics, normStats) {
  const scale = normStats.pwr.max / 100; // rough scaling
  return {
    rmse: metrics.rmse * scale,
    mae:  metrics.mae  * scale,
    mape: metrics.mape, // % stays same
  };
}

window.computeRMSE  = computeRMSE;
window.computeMAE   = computeMAE;
window.computeMAPE  = computeMAPE;
window.computeR2    = computeR2;
window.computeBaselinePredictions = computeBaselinePredictions;
window.PAPER_BENCHMARKS = PAPER_BENCHMARKS;
window.scaleMetricsToDataset = scaleMetricsToDataset;
