// ================================================================
// app.js — ORION–AURORA Interactive Demo
// React app with full pipeline visualization
// ================================================================

const { useState, useEffect, useRef, useCallback, useMemo } = React;
const {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend, AreaChart, Area
} = Recharts;

// ---- Colors ----
const C = {
  orion:   '#3b9eff',
  aurora:  '#a855f7',
  actual:  '#4ade80',
  baseline:'#6b7280',
  gold:    '#fbbf24',
  bg:      '#0d1117',
  surface: '#161b22',
  border:  '#21262d',
};

// ---- Custom tooltip ----
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-card">
      <div className="tooltip-label">t = {label}</div>
      {payload.map((p, i) => (
        <div className="tooltip-row" key={i}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span style={{ color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>
            {typeof p.value === 'number' ? p.value.toFixed(1) : p.value} kW
          </span>
        </div>
      ))}
    </div>
  );
}

// ---- Reservoir Heatmap ----
function ReservoirHeatmap({ states, label }) {
  const COLS = 25, ROWS = 4;
  const displayCount = COLS * ROWS;
  const cells = states.slice(0, displayCount);
  const maxAbs = Math.max(...cells.map(Math.abs), 0.01);

  function cellColor(v) {
    const n = v / maxAbs; // -1 to 1
    if (n > 0) {
      const h = 210, s = 80, l = 20 + n * 50;
      return `hsl(${h},${s}%,${l}%)`;
    } else {
      const h = 270, s = 70, l = 15 + Math.abs(n) * 40;
      return `hsl(${h},${s}%,${l}%)`;
    }
  }

  return (
    <div>
      <div className="heatmap-wrap">
        <div className="heatmap-grid" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }}>
          {cells.map((v, i) => (
            <div
              key={i}
              className="heatmap-cell"
              style={{ backgroundColor: cellColor(v) }}
              title={`Neuron ${i}: ${v.toFixed(4)}`}
            />
          ))}
        </div>
      </div>
      <div className="heatmap-scale">
        <span>−1</span>
        <div className="heatmap-scale-bar" />
        <span>+1</span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
        {label} · {cells.length} neurons displayed · Blue = positive activation · Purple = negative
      </div>
    </div>
  );
}

// ---- Pipeline Stage ----
function PipelineStage({ icon, name, sub, stage, activeStage, stageKey, extraClass = '' }) {
  const isActive = activeStage === stageKey || (activeStage === 'done' && true);
  const isDone   = activeStage === 'done';
  return (
    <div className="pipeline-stage">
      <div className={`pipeline-box stage-${stage} ${isActive && !isDone ? 'active' : ''} ${isDone ? `active ${extraClass}` : ''}`}>
        <div className="stage-icon">{icon}</div>
        <div className="stage-name">{name}</div>
        <div className="stage-sub">{sub}</div>
      </div>
    </div>
  );
}

// ---- Metric card ----
function MetricCard({ label, modelLabel, value, unit, delta, deltaLabel, variant }) {
  const cls = `metric-card mc-${variant}`;
  const labelCls = `metric-model ${variant}-label`;
  const deltaCls = delta < 0 ? 'delta-good' : 'delta-bad';
  return (
    <div className={cls}>
      <div className="metric-label">{label}</div>
      <div className={labelCls}>{modelLabel}</div>
      <div className="metric-value">
        {typeof value === 'number' ? value.toFixed(2) : '--'}
        <span className="metric-unit">{unit}</span>
      </div>
      {delta !== undefined && (
        <div className={`metric-delta ${deltaCls}`}>
          {delta < 0 ? '▼' : '▲'} {Math.abs(delta).toFixed(2)} {unit}
          <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>{deltaLabel}</span>
        </div>
      )}
    </div>
  );
}

// ---- Main App ----
function App() {
  // State
  const [dataset, setDataset]           = useState('uci');
  const [modelStatus, setModelStatus]   = useState('idle'); // idle | training | done | error
  const [pipelineStage, setPipelineStage] = useState('idle');
  const [progress, setProgress]         = useState(0);
  const [results, setResults]           = useState(null);
  const [liveInputs, setLiveInputs]     = useState({
    irr: 300, tmp: 12, hum: 65, wnd: 3, hour: 12, cc: 0.3
  });
  const [livePred, setLivePred]         = useState(null);
  const [reservoirStates, setReservoirStates] = useState(null);
  const [statusMsg, setStatusMsg]       = useState('Ready — click Run Model to begin');

  const orionRef  = useRef(null);
  const auroraRef = useRef(null);
  const normRef   = useRef(null);

  // ---- Data ----
  const rawData = useMemo(() => {
    return dataset === 'uci' ? window.UCI_DATA : window.AEP_DATA;
  }, [dataset]);

  const normStats = useMemo(() => {
    if (!rawData) return null;
    return window.computeNormStats(rawData);
  }, [rawData]);

  // ---- Train & Run Model ----
  const runModel = useCallback(async () => {
    if (modelStatus === 'training') return;
    setModelStatus('training');
    setProgress(0);
    setResults(null);
    setStatusMsg('Initializing ORION reservoir…');

    // Use setTimeout to yield to renderer between stages
    await new Promise(r => setTimeout(r, 50));

    const data = rawData;
    const ns   = normStats;
    normRef.current = ns;

    // Stage 1 — ORION
    setPipelineStage('input');
    setProgress(10);
    setStatusMsg('Building Echo State Network reservoir (40 neurons, SR=0.9)…');
    await new Promise(r => setTimeout(r, 80));

    const orion = new window.ORIONEngine({
      N: 40, F: 6, sr: 0.9,
      a: 0.3, beta: 0.05, lambda: 1e-3
    });
    orionRef.current = orion;

    setPipelineStage('orion');
    setProgress(30);
    setStatusMsg('ORION: Running reservoir state collection…');
    await new Promise(r => setTimeout(r, 80));

    // Split 80/20 train/test
    const splitIdx = Math.floor(data.length * 0.8);
    const trainData = data.slice(0, splitIdx);
    const testData  = data.slice(splitIdx);

    orion.train(trainData, ns);

    setProgress(55);
    setStatusMsg('ORION: Generating initial predictions via ridge regression readout…');
    await new Promise(r => setTimeout(r, 60));

    const orionAllPreds  = orion.predict(data, ns);
    const orionTestPreds = orionAllPreds.slice(splitIdx);

    // Capture reservoir snapshot
    setReservoirStates(orion.getSnapshot());

    // Stage 2 — AURORA
    setPipelineStage('aurora');
    setProgress(65);
    setStatusMsg('AURORA: Training adaptive correction layer…');
    await new Promise(r => setTimeout(r, 80));

    const aurora = new window.AURORAEngine({ windowSize: 24, radiantScaleFactor: 0.7 });
    auroraRef.current = aurora;

    aurora.train(trainData, orionAllPreds.slice(0, splitIdx), ns);

    setProgress(80);
    setStatusMsg('AURORA: Applying radiant adaptation module…');
    await new Promise(r => setTimeout(r, 80));

    const auroraAllPreds  = aurora.predict(data, orionAllPreds, ns);
    const auroraTestPreds = auroraAllPreds.slice(splitIdx);

    // Baseline
    const baselinePreds = window.computeBaselinePredictions(testData);
    const actuals       = testData.map(d => d.pwr);

    // Metrics
    setPipelineStage('output');
    setProgress(92);
    setStatusMsg('Computing RMSE, MAE, MAPE metrics…');
    await new Promise(r => setTimeout(r, 80));

    const metrics = {
      baseline: {
        rmse: window.computeRMSE(actuals, baselinePreds),
        mae:  window.computeMAE(actuals, baselinePreds),
        mape: window.computeMAPE(actuals, baselinePreds),
        r2:   window.computeR2(actuals, baselinePreds),
      },
      orion: {
        rmse: window.computeRMSE(actuals, orionTestPreds),
        mae:  window.computeMAE(actuals, orionTestPreds),
        mape: window.computeMAPE(actuals, orionTestPreds),
        r2:   window.computeR2(actuals, orionTestPreds),
      },
      aurora: {
        rmse: window.computeRMSE(actuals, auroraTestPreds),
        mae:  window.computeMAE(actuals, auroraTestPreds),
        mape: window.computeMAPE(actuals, auroraTestPreds),
        r2:   window.computeR2(actuals, auroraTestPreds),
      }
    };

    // Build chart data (show all test points, sampled for perf)
    const CHART_MAX = 120;
    const step  = Math.max(1, Math.floor(testData.length / CHART_MAX));
    const chartData = testData
      .filter((_, i) => i % step === 0)
      .map((row, i) => {
        const ti = i * step;
        return {
          idx:      i,
          label:    `${row.h}h d${row.doy}`,
          actual:   Math.round(actuals[ti]),
          orion:    Math.round(orionTestPreds[ti]),
          aurora:   Math.round(auroraTestPreds[ti]),
          baseline: Math.round(baselinePreds[ti]),
        };
      });

    // AEP grid contribution (only for AEP)
    let gridData = null;
    if (dataset === 'aep') {
      const recentData = testData.slice(-24);
      const avgSolar = auroraTestPreds.slice(-24).reduce((s, v) => s + v, 0) / 24 / 1000;
      const avgDemand = recentData.reduce((s, d) => s + (d.dem || 10000), 0) / 24 / 1000;
      gridData = {
        solarMW:   avgSolar,
        demandMW:  avgDemand,
        pct:       Math.min(100, (avgSolar / (avgDemand || 1)) * 100),
        otherMW:   Math.max(0, avgDemand - avgSolar),
      };
    } else {
      // UCI — estimate based on typical grid
      const avgSolar = auroraTestPreds.slice(-24).reduce((s, v) => s + v, 0) / 24 / 1000;
      const avgDemand = 8.0; // typical 8 GW
      gridData = {
        solarMW:  avgSolar,
        demandMW: avgDemand,
        pct:      Math.min(100, (avgSolar / avgDemand) * 100),
        otherMW:  Math.max(0, avgDemand - avgSolar),
      };
    }

    setResults({ metrics, chartData, gridData, splitIdx, data, orionAllPreds, auroraAllPreds });
    setPipelineStage('done');
    setProgress(100);
    setModelStatus('done');
    setStatusMsg(`Model complete · Test set: ${testData.length} samples · AURORA RMSE: ${metrics.aurora.rmse.toFixed(2)} kW`);

  }, [rawData, normStats, dataset, modelStatus]);

  // ---- Live prediction from sliders ----
  useEffect(() => {
    if (!orionRef.current || !auroraRef.current || !normRef.current) return;
    const ns = normRef.current;
    const orion = orionRef.current;
    const aurora = auroraRef.current;

    const n = (v, k) => (ns[k].max - ns[k].min) > 0 ? (v - ns[k].min) / (ns[k].max - ns[k].min) : 0;
    const features = [
      n(liveInputs.irr, 'irr'), n(liveInputs.tmp, 'tmp'),
      n(liveInputs.hum, 'hum'), n(liveInputs.wnd, 'wnd'),
      Math.sin(2 * Math.PI * liveInputs.hour / 24),
      Math.cos(2 * Math.PI * liveInputs.hour / 24),
    ];

    const cloudFactor = Math.max(0, 1 - liveInputs.cc * 0.8);
    const orionPred  = orion.predictOne(features, ns, liveInputs.hour) * cloudFactor;
    // AURORA feat8: [orion_norm, irr_norm, tmp_norm, hum_norm, wnd_norm, sin, cos, 1]
    const auroraFeatures = [orionPred / (ns.pwr.max || 1), ...features, 1.0];
    const auroraPred = aurora.predictOne(auroraFeatures, orionPred, ns);

    setLivePred({ orion: orionPred, aurora: Math.max(0, auroraPred) });
  }, [liveInputs, modelStatus]);

  // ---- Dataset switch resets model ----
  useEffect(() => {
    setModelStatus('idle');
    setPipelineStage('idle');
    setProgress(0);
    setResults(null);
    setLivePred(null);
    setReservoirStates(null);
    setStatusMsg('Dataset changed — click Run Model');
    orionRef.current  = null;
    auroraRef.current = null;
    normRef.current   = null;
  }, [dataset]);

  // ---- CSV Export ----
  function exportCSV() {
    if (!results) return;
    const rows = results.chartData.map(d =>
      `${d.label},${d.actual},${d.orion},${d.aurora},${d.baseline}`
    );
    const csv = 'timestamp,actual_kw,orion_kw,aurora_kw,baseline_kw\n' + rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `orion_aurora_${dataset}_predictions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---- Paper targets for dataset ----
  const targets = dataset === 'uci'
    ? { rmse: 3.50, mae: 2.00, mape: 1.50 }
    : { rmse: 4.00, mae: 2.50, mape: 2.00 };

  const benchmarks = window.PAPER_BENCHMARKS[dataset];

  // ---- Donut chart data for grid panel ----
  const donutData = results ? [
    { name: 'Solar', value: Math.round(results.gridData.pct) },
    { name: 'Other', value: Math.round(100 - results.gridData.pct) },
  ] : [];
  const DONUT_COLORS = [C.aurora, '#1e2a3a'];

  // ---- Slider config ----
  const sliders = [
    { key: 'irr',  label: 'Solar Irradiance',  unit: 'W/m²', min: 0,   max: 1000, step: 5 },
    { key: 'tmp',  label: 'Temperature',        unit: '°C',   min: -10, max: 40,   step: 0.5 },
    { key: 'hum',  label: 'Humidity',           unit: '%',    min: 0,   max: 100,  step: 1 },
    { key: 'wnd',  label: 'Wind Speed',         unit: 'm/s',  min: 0,   max: 20,   step: 0.5 },
    { key: 'hour', label: 'Hour of Day',        unit: 'h',    min: 0,   max: 23,   step: 1 },
    { key: 'cc',   label: 'Cloud Cover',        unit: '',     min: 0,   max: 1,    step: 0.01 },
  ];

  const improvement = results
    ? ((results.metrics.orion.rmse - results.metrics.aurora.rmse) / results.metrics.orion.rmse * 100).toFixed(1)
    : null;

  return (
    <div>
      {/* ---- Header ---- */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">☀️</div>
          <div>
            <div className="logo-text">
              <span className="orion">ORION</span>
              <span className="sep">–</span>
              <span className="aurora">AURORA</span>
            </div>
            <div className="header-subtitle">Dual-Model Solar Power Forecasting System</div>
          </div>
        </div>
        <div className="header-actions">
          <div className="dataset-toggle">
            <button id="btn-uci" className={dataset === 'uci' ? 'active' : ''} onClick={() => setDataset('uci')}>
              UCI Solar
            </button>
            <button id="btn-aep" className={dataset === 'aep' ? 'active' : ''} onClick={() => setDataset('aep')}>
              AEP Grid
            </button>
          </div>
          <button className="btn-export" onClick={exportCSV} disabled={!results} id="btn-export">
            ⬇ Export CSV
          </button>
          <button
            id="btn-run-model"
            className={`btn-run ${modelStatus === 'training' ? 'running' : ''}`}
            onClick={runModel}
            disabled={modelStatus === 'training'}
          >
            {modelStatus === 'training' ? '⚡ Running…' : modelStatus === 'done' ? '↺ Re-Run' : '▶ Run Model'}
          </button>
        </div>
      </header>

      {/* ---- Main ---- */}
      <main className="main">

        {/* ---- Status bar ---- */}
        <div className="status-bar">
          <div className={`status-indicator ${modelStatus === 'training' ? 'running' : modelStatus === 'done' ? '' : 'idle'}`} />
          <span>{statusMsg}</span>
          {modelStatus === 'training' && (
            <div style={{ flex: 1, marginLeft: 12 }}>
              <div className="progress-bar-wrap">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
          {modelStatus === 'done' && improvement && (
            <span style={{ marginLeft: 'auto', color: 'var(--green)', fontSize: 11, fontWeight: 600 }}>
              AURORA improved RMSE by {improvement}% vs ORION alone
            </span>
          )}
        </div>

        {/* ---- Pipeline Visualization ---- */}
        <div className="card fade-in">
          <div className="card-header">
            <div className="card-title">
              <div className="dot dot-orion" />
              Pipeline Architecture
            </div>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {dataset === 'uci' ? 'UCI Solar Dataset' : 'AEP Grid Dataset'} · {rawData?.length} samples
            </span>
          </div>

          <div className="pipeline-wrap">
            {/* Input */}
            <div className="pipeline-stage">
              <div className={`pipeline-box stage-input ${['input','orion','aurora','output','done'].includes(pipelineStage) ? 'active' : ''}`}>
                <div className="stage-icon">📡</div>
                <div className="stage-name">Raw Input</div>
                <div className="stage-sub">irr·tmp·hum·wnd·h</div>
              </div>
              <div className="pipeline-label">6 features + normalization</div>
            </div>

            <div className={`pipeline-arrow ${['orion','aurora','output','done'].includes(pipelineStage) ? 'lit' : ''}`}>→</div>

            {/* ORION */}
            <div className="pipeline-stage">
              <div className={`pipeline-box stage-orion ${['orion','aurora','output','done'].includes(pipelineStage) ? 'active' : ''}`}>
                <div className="stage-icon">🔵</div>
                <div className="stage-name">ORION</div>
                <div className="stage-sub">ESN·TFT·Readout</div>
              </div>
              <div className="pipeline-label">100-neuron reservoir + ridge regression</div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, marginTop: -18 }}>
              <div className={`pipeline-arrow ${['aurora','output','done'].includes(pipelineStage) ? 'lit' : ''}`}>→</div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', textAlign: 'center', maxWidth: 80 }}>
                Initial forecast + reservoir states
              </div>
            </div>

            {/* AURORA */}
            <div className="pipeline-stage">
              <div className={`pipeline-box stage-aurora ${['aurora','output','done'].includes(pipelineStage) ? 'active aurora-active' : ''}`}>
                <div className="stage-icon">🟣</div>
                <div className="stage-name">AURORA</div>
                <div className="stage-sub">Adaptive·Radiant</div>
              </div>
              <div className="pipeline-label">Adaptive gain + windowed correction</div>
            </div>

            <div className={`pipeline-arrow ${['output','done'].includes(pipelineStage) ? 'lit' : ''}`}>→</div>

            {/* Output */}
            <div className="pipeline-stage">
              <div className={`pipeline-box stage-output ${['output','done'].includes(pipelineStage) ? 'active output-active' : ''}`}>
                <div className="stage-icon">⚡</div>
                <div className="stage-name">Forecast</div>
                <div className="stage-sub">kW · ±error</div>
              </div>
              <div className="pipeline-label">Final solar PV power prediction</div>
            </div>
          </div>

          {/* Stage info */}
          <div className="stage-info-grid" style={{ marginTop: 20 }}>
            <div className="stage-info-box orion-box">
              <div className="stage-info-title orion-col">⚙ ORION Components</div>
              <div className="stage-info-item">Echo State Network — 100×100 reservoir, SR=0.9</div>
              <div className="stage-info-item">Oscillatory modulation β·sin(ωt+φ) for cyclic patterns</div>
              <div className="stage-info-item">Leaking rate α=0.3 for temporal smoothing</div>
              <div className="stage-info-item">Ridge regression readout (λ=1e-4) — closed-form</div>
              <div className="stage-info-item">Captures short-term spikes + daily arcs</div>
            </div>
            <div className="stage-info-box aurora-box">
              <div className="stage-info-title aurora-col">⚙ AURORA Components</div>
              <div className="stage-info-item">Adaptive gain correction layer (8 features + bias)</div>
              <div className="stage-info-item">Radiant adaptation: windowed variance scaling</div>
              <div className="stage-info-item">24-hour rolling window for local uncertainty estimate</div>
              <div className="stage-info-item">Corrects for non-stationarity & weather regime drift</div>
              <div className="stage-info-item">Improves ORION output at cloud burst onset, dawn/dusk</div>
            </div>
          </div>
        </div>

        {/* ---- Live Input Panel ---- */}
        <div className="card fade-in">
          <div className="card-header">
            <div className="card-title">
              <div className="dot dot-green" />
              Live Input Panel
              {modelStatus !== 'done' && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>
                  (Run the model first to enable predictions)
                </span>
              )}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Adjust sliders → predictions update in real-time
            </div>
          </div>

          <div style={{ display: 'flex', gap: 24, alignItems: 'stretch' }}>
            <div className="sliders-grid" style={{ flex: 1 }}>
              {sliders.map(s => (
                <div className="slider-row" key={s.key} id={`slider-${s.key}`}>
                  <label>
                    {s.label}
                    <span>{liveInputs[s.key]}{s.unit}</span>
                  </label>
                  <input
                    type="range"
                    min={s.min} max={s.max} step={s.step}
                    value={liveInputs[s.key]}
                    onChange={e => setLiveInputs(prev => ({ ...prev, [s.key]: parseFloat(e.target.value) }))}
                  />
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minWidth: 200 }}>
              <div className="live-pred-wrap">
                <div className="live-pred-label">ORION Initial Forecast</div>
                <div className="live-orion-val">
                  {livePred ? livePred.orion.toFixed(0) : '---'}
                  <span style={{ fontSize: 14, color: 'var(--text-muted)', marginLeft: 4 }}>kW</span>
                </div>
              </div>
              <div className="live-pred-wrap" style={{ borderColor: 'hsl(270,50%,30%)' }}>
                <div className="live-pred-label">AURORA Refined Forecast</div>
                <div className="live-aurora-val">
                  {livePred ? livePred.aurora.toFixed(0) : '---'}
                  <span style={{ fontSize: 14, color: 'var(--text-muted)', marginLeft: 4 }}>kW</span>
                </div>
                {livePred && livePred.aurora !== livePred.orion && (
                  <div className="live-improvement-badge">
                    {livePred.aurora > livePred.orion
                      ? `+${(livePred.aurora - livePred.orion).toFixed(0)} kW correction`
                      : `${(livePred.aurora - livePred.orion).toFixed(0)} kW correction`}
                  </div>
                )}
              </div>
              <div style={{
                padding: '12px 16px', background: 'var(--bg-surface)',
                border: '1px solid var(--border)', borderRadius: 12,
                fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6
              }}>
                <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>Feature Vector</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
                  irr={liveInputs.irr}<br/>
                  tmp={liveInputs.tmp}°C · hum={liveInputs.hum}%<br/>
                  wnd={liveInputs.wnd} m/s · cc={liveInputs.cc}<br/>
                  h_sin={Math.sin(2*Math.PI*liveInputs.hour/24).toFixed(3)}<br/>
                  h_cos={Math.cos(2*Math.PI*liveInputs.hour/24).toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ---- Results (only after model run) ---- */}
        {results && (
          <>
            {/* ---- Metrics Dashboard ---- */}
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">
                  <div className="dot dot-gold" />
                  Metrics Dashboard
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Test set · {Math.floor(rawData.length * 0.2)} samples · vs paper targets
                </div>
              </div>

              <div className="grid-3" style={{ marginBottom: 20 }}>
                {/* RMSE */}
                <div>
                  <div className="section-label">RMSE (kW)</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <MetricCard label="RMSE" modelLabel="Baseline (Mean)" value={results.metrics.baseline.rmse} unit="kW" variant="baseline" />
                    <MetricCard
                      label="RMSE" modelLabel="ORION Stage 1" value={results.metrics.orion.rmse} unit="kW"
                      delta={results.metrics.orion.rmse - results.metrics.baseline.rmse} deltaLabel="vs baseline"
                      variant="orion"
                    />
                    <MetricCard
                      label="RMSE" modelLabel="ORION + AURORA" value={results.metrics.aurora.rmse} unit="kW"
                      delta={results.metrics.aurora.rmse - results.metrics.orion.rmse} deltaLabel="vs ORION"
                      variant="aurora"
                    />
                  </div>
                </div>
                {/* MAE */}
                <div>
                  <div className="section-label">MAE (kW)</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <MetricCard label="MAE" modelLabel="Baseline (Mean)" value={results.metrics.baseline.mae} unit="kW" variant="baseline" />
                    <MetricCard
                      label="MAE" modelLabel="ORION Stage 1" value={results.metrics.orion.mae} unit="kW"
                      delta={results.metrics.orion.mae - results.metrics.baseline.mae} deltaLabel="vs baseline"
                      variant="orion"
                    />
                    <MetricCard
                      label="MAE" modelLabel="ORION + AURORA" value={results.metrics.aurora.mae} unit="kW"
                      delta={results.metrics.aurora.mae - results.metrics.orion.mae} deltaLabel="vs ORION"
                      variant="aurora"
                    />
                  </div>
                </div>
                {/* MAPE */}
                <div>
                  <div className="section-label">MAPE (%)</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <MetricCard label="MAPE" modelLabel="Baseline (Mean)" value={results.metrics.baseline.mape} unit="%" variant="baseline" />
                    <MetricCard
                      label="MAPE" modelLabel="ORION Stage 1" value={results.metrics.orion.mape} unit="%"
                      delta={results.metrics.orion.mape - results.metrics.baseline.mape} deltaLabel="vs baseline"
                      variant="orion"
                    />
                    <MetricCard
                      label="MAPE" modelLabel="ORION + AURORA" value={results.metrics.aurora.mape} unit="%"
                      delta={results.metrics.aurora.mape - results.metrics.orion.mape} deltaLabel="vs ORION"
                      variant="aurora"
                    />
                  </div>
                </div>
              </div>

              {/* Paper targets comparison */}
              <div style={{
                padding: '14px 18px', borderRadius: 12,
                background: 'hsl(45,50%,30%,0.08)', border: '1px solid hsl(45,50%,30%,0.2)',
                display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'center'
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--gold)', letterSpacing: 1, textTransform: 'uppercase' }}>
                  📄 Paper Targets
                </div>
                {[['RMSE', targets.rmse, 'kW'], ['MAE', targets.mae, 'kW'], ['MAPE', targets.mape, '%']].map(([k, v, u]) => (
                  <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{k}</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: 'var(--gold)' }}>
                      {v}{u}
                    </div>
                  </div>
                ))}
                <div style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-secondary)' }}>
                  Live-computed values approximate paper results using a {rawData.length}-sample subset
                </div>
              </div>
            </div>

            {/* ---- Actual vs Predicted Chart ---- */}
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">
                  <div className="dot dot-green" />
                  Actual vs Predicted — Test Set
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Showing {results.chartData.length} sampled test-set predictions
                </div>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={results.chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradActual"  x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor={C.actual}  stopOpacity={0.15} />
                        <stop offset="95%" stopColor={C.actual}  stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="gradAurora"  x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor={C.aurora}  stopOpacity={0.12} />
                        <stop offset="95%" stopColor={C.aurora}  stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(224,20%,14%)" />
                    <XAxis dataKey="idx" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} label={{ value: 'Time Index', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `${(v/1000).toFixed(1)}MW`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="actual"   name="Actual"       stroke={C.actual}   strokeWidth={2} fill="url(#gradActual)"  dot={false} />
                    <Area type="monotone" dataKey="aurora"   name="AURORA Final" stroke={C.aurora}   strokeWidth={2} fill="url(#gradAurora)"  dot={false} strokeDasharray="0" />
                    <Line type="monotone" dataKey="orion"    name="ORION Init"   stroke={C.orion}    strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="baseline" name="Baseline"     stroke={C.baseline} strokeWidth={1} dot={false} strokeDasharray="2 4" opacity={0.6} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-legend">
                {[
                  { color: C.actual,   label: 'Actual Power Output' },
                  { color: C.aurora,   label: 'AURORA Final Prediction' },
                  { color: C.orion,    label: 'ORION Initial Prediction' },
                  { color: C.baseline, label: 'Baseline (Mean)' },
                ].map(({ color, label }) => (
                  <div className="legend-item" key={label}>
                    <div className="legend-dot" style={{ background: color }} />
                    {label}
                  </div>
                ))}
              </div>
            </div>

            {/* ---- Benchmark comparison + Reservoir heatmap ---- */}
            <div className="grid-2">
              {/* Benchmark bar chart */}
              <div className="card fade-in">
                <div className="card-header">
                  <div className="card-title">
                    <div className="dot dot-gold" />
                    Model Comparison (RMSE)
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Paper benchmark values</div>
                </div>
                <div style={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={benchmarks.map(b => ({
                        name: b.model.replace(' only', '').replace(' + AURORA', '+AUR'),
                        rmse: b.rmse,
                        isOrion: b.isOrion,
                        isAurora: b.isAurora,
                      }))}
                      layout="vertical"
                      margin={{ top: 4, right: 20, left: 0, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(224,20%,14%)" />
                      <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} domain={[0, 12]} />
                      <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} tickLine={false} axisLine={false} width={90} />
                      <Tooltip formatter={(v) => [`${v} kW`, 'RMSE']} contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8 }} />
                      <Bar dataKey="rmse" radius={[0, 4, 4, 0]}>
                        {benchmarks.map((b, i) => (
                          <Cell
                            key={i}
                            fill={b.isAurora ? C.aurora : b.isOrion ? C.orion : 'hsl(224,20%,22%)'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ marginTop: 12 }}>
                  <table className="bench-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>MAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarks.slice(-4).map((b, i) => (
                        <tr key={i} className={b.isAurora ? 'highlight-aurora' : b.isOrion ? 'highlight' : ''}>
                          <td className="model-name-cell">
                            {b.model}
                            {b.isOrion  && <span className="badge-orion">STAGE 1</span>}
                            {b.isAurora && <span className="badge-aurora">FULL</span>}
                            {b.isAurora && <span className="badge-best">BEST</span>}
                          </td>
                          <td>{b.rmse}</td>
                          <td>{b.mae}</td>
                          <td>{b.mape}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Reservoir heatmap */}
              <div className="card fade-in">
                <div className="card-header">
                  <div className="card-title">
                    <div className="dot dot-orion" />
                    ORION Reservoir State Monitor
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>100-neuron ESN activation snapshot</div>
                </div>
                {reservoirStates && reservoirStates.length > 0 ? (
                  <ReservoirHeatmap states={reservoirStates} label="ORION ESN" />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 32 }}>
                    Run the model to see reservoir activations
                  </div>
                )}

                <div style={{ marginTop: 20 }}>
                  <div className="section-label" style={{ marginBottom: 12 }}>R² Score Comparison</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {[
                      { label: 'Baseline', val: results.metrics.baseline.r2, color: C.baseline },
                      { label: 'ORION',    val: results.metrics.orion.r2,    color: C.orion },
                      { label: 'AURORA',   val: results.metrics.aurora.r2,   color: C.aurora },
                    ].map(({ label, val, color }) => (
                      <div key={label}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                          <span style={{ fontFamily: 'JetBrains Mono, monospace', color }}>{val.toFixed(4)}</span>
                        </div>
                        <div style={{ height: 6, background: 'var(--bg-deep)', borderRadius: 3 }}>
                          <div style={{
                            height: '100%', borderRadius: 3,
                            width: `${Math.max(0, Math.min(100, val * 100))}%`,
                            background: color,
                            transition: 'width 0.8s ease',
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* ---- Grid Contribution ---- */}
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">
                  <div className="dot dot-aurora" />
                  Solar Grid Contribution Panel
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Based on AURORA predicted output · Last 24h average
                </div>
              </div>

              <div className="grid-panel">
                <div className="grid-donut-wrap">
                  <PieChart width={180} height={180}>
                    <Pie
                      data={donutData}
                      cx={90} cy={90}
                      innerRadius={55} outerRadius={80}
                      startAngle={90} endAngle={-270}
                      dataKey="value"
                      strokeWidth={0}
                    >
                      {donutData.map((_, i) => (
                        <Cell key={i} fill={DONUT_COLORS[i]} />
                      ))}
                    </Pie>
                  </PieChart>
                  <div className="grid-donut-center">
                    <div className="grid-pct">{results.gridData.pct.toFixed(1)}%</div>
                    <div className="grid-pct-label">Solar Share</div>
                  </div>
                </div>

                <div className="grid-stats">
                  {[
                    {
                      label: 'Predicted Solar Output',
                      value: `${results.gridData.solarMW.toFixed(2)} MW`,
                      pct: results.gridData.pct,
                      color: C.aurora,
                    },
                    {
                      label: 'Total Grid Demand',
                      value: `${results.gridData.demandMW.toFixed(2)} MW`,
                      pct: 100,
                      color: C.baseline,
                    },
                    {
                      label: 'Non-Solar Generation',
                      value: `${results.gridData.otherMW.toFixed(2)} MW`,
                      pct: 100 - results.gridData.pct,
                      color: '#4b5563',
                    },
                  ].map(({ label, value, pct, color }) => (
                    <div className="grid-stat-row" key={label}>
                      <div className="grid-stat-label">{label}</div>
                      <div className="grid-stat-value" style={{ color }}>{value}</div>
                      <div className="grid-stat-bar-wrap">
                        <div className="grid-stat-bar" style={{ width: `${Math.min(100, pct)}%`, background: color }} />
                      </div>
                    </div>
                  ))}
                  <div style={{
                    padding: '10px 14px', borderRadius: 8, background: 'hsl(270,50%,25%,0.15)',
                    border: '1px solid hsl(270,50%,35%,0.3)', fontSize: 11, color: 'var(--text-secondary)',
                    lineHeight: 1.6
                  }}>
                    <strong style={{ color: 'var(--aurora)' }}>AURORA forecast insight:</strong> At{' '}
                    {results.gridData.pct.toFixed(1)}% solar penetration, grid operators can{' '}
                    {results.gridData.pct > 15 ? 'significantly reduce' : 'partially offset'}{' '}
                    conventional generation dispatch for the next 24 hours.
                    RMSE of {results.metrics.aurora.rmse.toFixed(2)} kW provides ±{(results.metrics.aurora.rmse * 1.96).toFixed(2)} kW
                    95% confidence bounds for scheduling.
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* ---- Call to action if not run yet ---- */}
        {modelStatus === 'idle' && (
          <div className="card fade-in" style={{ textAlign: 'center', padding: '48px 24px' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>☀️</div>
            <h2 className="text-gradient" style={{ fontSize: 28, fontWeight: 800, marginBottom: 12 }}>
              Ready to Forecast
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: 14, maxWidth: 480, margin: '0 auto 24px' }}>
              Click <strong style={{ color: 'var(--orion)' }}>Run Model</strong> to train the ORION Echo State Network and AURORA
              adaptive refinement layer on the {dataset === 'uci' ? 'UCI Solar' : 'AEP Grid'} dataset.
              Predictions compute entirely in-browser — no server required.
            </p>
            <button className="btn-run" style={{ fontSize: 15, padding: '12px 32px', display: 'inline-flex' }} onClick={runModel}>
              ▶ Run ORION–AURORA
            </button>
          </div>
        )}

      </main>

      {/* ---- Footer ---- */}
      <footer style={{
        borderTop: '1px solid var(--border)', padding: '16px 32px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        color: 'var(--text-muted)', fontSize: 11,
      }}>
        <span>
          <strong style={{ color: 'var(--text-secondary)' }}>ORION–AURORA</strong> · Dual-Model Solar Forecasting Demo ·
          Browser-native ESN + Adaptive Refinement
        </span>
        <span>
          Based on: <em>"Dual-Model Deep Learning Framework with ORION and AURORA for Robust and Accurate Solar Power Prediction in Smart Grid Systems"</em>
        </span>
      </footer>
    </div>
  );
}

// ---- Error Boundary ----
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(e) { return { error: e }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{padding:40,fontFamily:'monospace',color:'#f87171',background:'#0d1117',minHeight:'100vh'}}>
          <h2 style={{color:'#fbbf24',marginBottom:16}}>⚠ ORION-AURORA Error</h2>
          <pre style={{whiteSpace:'pre-wrap',fontSize:13}}>{String(this.state.error)}</pre>
          <pre style={{whiteSpace:'pre-wrap',fontSize:11,color:'#6b7280',marginTop:12}}>{this.state.error?.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

// ---- Mount ----
try {
  const rootEl = document.getElementById('root');
  const root = ReactDOM.createRoot(rootEl);
  root.render(<ErrorBoundary><App /></ErrorBoundary>);
  console.log('[ORION] React mounted successfully');
} catch(e) {
  console.error('[ORION] Mount failed:', e);
  document.getElementById('root').innerHTML =
    '<div style="padding:40px;font-family:monospace;color:#f87171;background:#0d1117;min-height:100vh">' +
    '<h2 style="color:#fbbf24">Mount Error</h2><pre>' + String(e) + '</pre></div>';
}
