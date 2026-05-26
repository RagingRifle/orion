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
  const [activeTab, setActiveTab]       = useState('sandbox'); // sandbox | live
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

  // ---- API Configuration & Location States ----
  const [weatherKey, setWeatherKey]     = useState(() => localStorage.getItem('orion_weather_key') || '');
  const [geminiKey, setGeminiKey]       = useState(() => localStorage.getItem('orion_gemini_key') || '');
  const [lat, setLat]                   = useState(() => localStorage.getItem('orion_lat') || '36.1716'); // Default: Las Vegas
  const [lon, setLon]                   = useState(() => localStorage.getItem('orion_lon') || '-115.1398');

  // Key Visibility States
  const [showWeatherKey, setShowWeatherKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey]   = useState(false);

  // Live Weather & Prediction States
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [weatherError, setWeatherError]     = useState(null);
  const [weatherLocation, setWeatherLocation] = useState('');
  const [currentWeather, setCurrentWeather] = useState(null);
  const [liveForecastResults, setLiveForecastResults] = useState(null);

  // Gemini AI Advisor States
  const [aiReport, setAiReport]         = useState(null);
  const [aiLoading, setAiLoading]       = useState(false);
  const [aiError, setAiError]           = useState(null);

  // ---- Persistence Hooks ----
  useEffect(() => { localStorage.setItem('orion_weather_key', weatherKey); }, [weatherKey]);
  useEffect(() => { localStorage.setItem('orion_gemini_key', geminiKey); }, [geminiKey]);
  useEffect(() => { localStorage.setItem('orion_lat', lat); }, [lat]);
  useEffect(() => { localStorage.setItem('orion_lon', lon); }, [lon]);

  // ---- Hardcoded mock/demo AI advice fallback ----
  const getMockAiReport = useCallback(() => {
    return `### Solar Generation Synopsis
- **Peak Performance Expected**: Local solar generation is projected to peak at **928 kW** tomorrow at **12:00 PM**, driven by clear atmospheric windows and favorable temperatures.
- **Overcast Ingress Alert**: A rapid cloud cover surge (from **15% to 80%**) is forecasted for the afternoon of Day 2 between **2:00 PM and 5:00 PM**. This will cause solar output to plunge by **72%** within a 90-minute window, creating a steep generation ramp down.
- **Thermal Efficiency Benefit**: Favorable temperature winds at **22°C** will sustain optimum cell temperatures, preserving module efficiency near maximum capacity during the noon peak.

### Energy Storage & Battery Strategy
- **Pre-Charge Phase (08:00 - 10:00)**: Initiate battery energy storage system (BESS) pre-charging at a modest **100 kW/h** to absorb early-morning ramp-up.
- **Peak Surplus Absorption (11:00 - 14:00)**: Divert all solar surplus above **650 kW** to storage. Charge rate can be maximized up to **350 kW/h** to cap local substation thermal loading.
- **Evening Support Discharge (17:00 - 20:00)**: Discharge BESS at a steady **200 kW/h** rate to support the local grid during peak evening consumption and offset the sharp solar ramp-down.

### Grid Stability & Load Shifting
- **Load Matching Scheduling**: Schedule heavy industrial water-pumping and manufacturing operations to run strictly between **11:00 AM and 2:00 PM** to match peak local PV production.
- **Demand Response Trigger**: Prepare for a grid stability warning at **3:30 PM on Day 2** due to the sudden cloud-cover plunge. Throttling of non-essential cooling systems is recommended.
- **EV Station Optimization**: Enable fast-charging stations at discounted tariffs between **10:00 AM and 2:00 PM** to maximize localized solar consumption.

### Consumer Action Prompts
- **Pre-Cooling Guidance**: Residential smart thermostats should be set to pre-cool homes to **21°C** at **1:00 PM** using local solar, then reset to **24°C** at **4:00 PM** to minimize peak grid load.
- **Smart Appliance Delay**: Schedule pool filtration pumps, water heaters, and laundry cycles to run in the designated **10:30 AM - 3:00 PM** solar-abundance window.`;
  }, []);

  // ---- Dataset embeddings (UCI / AEP) ----
  const rawData = useMemo(() => {
    return dataset === 'uci' ? window.UCI_DATA : window.AEP_DATA;
  }, [dataset]);

  const normStats = useMemo(() => {
    if (!rawData) return null;
    return window.computeNormStats(rawData);
  }, [rawData]);

  // ---- Silent Background Training (for live forecast dependency) ----
  const ensureTrained = useCallback(async () => {
    if (orionRef.current && auroraRef.current && normRef.current) {
      return {
        orion: orionRef.current,
        aurora: auroraRef.current,
        normStats: normRef.current
      };
    }

    const data = rawData;
    const ns   = normStats;
    normRef.current = ns;

    const orion = new window.ORIONEngine({
      N: 40, F: 6, sr: 0.9,
      a: 0.3, beta: 0.05, lambda: 1e-3
    });
    orionRef.current = orion;

    // Split 80/20 train/test
    const splitIdx = Math.floor(data.length * 0.8);
    const trainData = data.slice(0, splitIdx);

    orion.train(trainData, ns);
    const orionAllPreds = orion.predict(data, ns);
    setReservoirStates(orion.getSnapshot());

    const aurora = new window.AURORAEngine({ windowSize: 24, radiantScaleFactor: 0.7 });
    auroraRef.current = aurora;
    aurora.train(trainData, orionAllPreds.slice(0, splitIdx), ns);

    return { orion, aurora, normStats: ns };
  }, [rawData, normStats]);

  // ---- Weather API Fetch & Prediction Engine ----
  const fetchLiveForecast = useCallback(async () => {
    if (!weatherKey) {
      setWeatherError('Please enter an OpenWeatherMap API Key in the settings bar.');
      return;
    }
    const cleanLat = parseFloat(lat);
    const cleanLon = parseFloat(lon);
    if (isNaN(cleanLat) || cleanLat < -90 || cleanLat > 90) {
      setWeatherError('Please enter a valid Latitude (-90 to 90).');
      return;
    }
    if (isNaN(cleanLon) || cleanLon < -180 || cleanLon > 180) {
      setWeatherError('Please enter a valid Longitude (-180 to 180).');
      return;
    }

    setWeatherLoading(true);
    setWeatherError(null);
    setAiReport(null);

    try {
      const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${cleanLat}&lon=${cleanLon}&appid=${weatherKey}&units=metric`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Weather API Error: ${response.status} ${response.statusText}. Please verify your API Key and internet connection.`);
      }
      const data = await response.json();
      
      const city = data.city?.name ? `${data.city.name}, ${data.city.country || ''}` : `Coordinates (${cleanLat}, ${cleanLon})`;
      setWeatherLocation(city);

      const list = data.list || [];
      if (!list.length) {
        throw new Error('No weather forecast entries returned from the API.');
      }

      // 1. Process forecasts into model features
      const forecastRows = list.map(item => {
        const date = new Date(item.dt * 1000);
        const hour = date.getHours();
        const solarHour = hour + date.getMinutes() / 60;
        
        // Solar Irradiance estimation: peak clear-sky ~950 W/m² at solar noon (12:00), zero at night (6:00 to 18:00)
        const clearIrr = Math.max(0, Math.sin(Math.PI * (solarHour - 6) / 12)) * 950;
        const cloudFraction = (item.clouds?.all || 0) / 100;
        // Irradiance falls off quadratically with cloud cover
        const estimatedIrr = clearIrr * (1 - 0.78 * (cloudFraction ** 2));

        return {
          dt: item.dt,
          h: hour,
          doy: 150, // Arbitrary warm day of year
          irr: estimatedIrr,
          tmp: item.main?.temp || 15,
          hum: item.main?.humidity || 50,
          wnd: item.wind?.speed || 2,
          cc: cloudFraction,
          timeStr: date.toLocaleTimeString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }),
          dateStr: date.toLocaleDateString([], { month: 'short', day: 'numeric' })
        };
      });

      // 2. Ensure models are trained and initialized
      const engines = await ensureTrained();
      const { orion, aurora, normStats: ns } = engines;

      // 3. Generate predictions sequentially to preserve Echo State dynamics
      const orionPreds = orion.predict(forecastRows, ns);
      const auroraPreds = aurora.predict(forecastRows, orionPreds, ns);

      // 4. Map back to chart models
      const chartData = forecastRows.map((row, i) => ({
        idx: i,
        label: row.timeStr,
        dateLabel: row.dateStr,
        temp: row.tmp,
        cloud: Math.round(row.cc * 100),
        irr: Math.round(row.irr),
        orion: Math.round(orionPreds[i]),
        aurora: Math.round(auroraPreds[i])
      }));

      setCurrentWeather(forecastRows[0]);
      setLiveForecastResults({ chartData, forecastRows, orionPreds, auroraPreds });

    } catch (err) {
      console.error('[WEATHER] Failed to load forecast:', err);
      setWeatherError(err.message || 'An unexpected error occurred while fetching the weather forecast.');
    } finally {
      setWeatherLoading(false);
    }
  }, [weatherKey, lat, lon, ensureTrained]);

  // ---- Gemini AI Grid Advisor Fetcher ----
  const generateAiAdvice = useCallback(async () => {
    if (!liveForecastResults) {
      setAiError('Please fetch the weather forecast first so the AI has data to analyze.');
      return;
    }

    setAiLoading(true);
    setAiError(null);
    setAiReport(null);

    // Fallback: If no Gemini key, show mock/demo report immediately with minor delay
    if (!geminiKey) {
      await new Promise(r => setTimeout(r, 900));
      setAiReport(getMockAiReport());
      setAiLoading(false);
      return;
    }

    try {
      const { chartData } = liveForecastResults;
      // Compile a concise forecast data block to keep prompt context focused
      const forecastSummary = chartData.slice(0, 16).map(d => 
        `- ${d.dateLabel} at ${d.label}: Clouds=${d.cloud}%, Irradiance=${d.irr} W/m², Temp=${d.temp}°C, ORION Pred=${d.orion} kW, AURORA Final=${d.aurora} kW`
      ).join('\n');

      const prompt = `You are a Smart Grid Operations Director managing a solar PV energy storage and grid integration hub.
The location of the solar array is: Latitude ${lat}, Longitude ${lon}.

Here is the dual-stage deep learning forecast (using ORION Echo State Networks and AURORA adaptive refinement) for the upcoming solar power output and weather conditions over the next 48 hours:

${forecastSummary}

Based on this forecast, please generate a highly professional, comprehensive "AI Grid Advisor Operations Report" containing:
1. ### Solar Generation Synopsis: A summary of the peak power expectations, seasonal trends, and the main weather factors (like cloud cover onset) affecting generation.
2. ### Energy Storage & Battery Strategy: Specific hour-by-hour instructions on when to charge grid-scale batteries (e.g. from solar peak surplus) and when to discharge/shave peaks.
3. ### Grid Stability & Load Shifting: Recommendations on commercial load deferral, industrial demand response actions, or EV charging coordination based on predicted dips.
4. ### Consumer Action Prompts: Friendly, actionable recommendations for local smart home users (e.g., HVAC pre-cooling or running appliances).

Format your output in clean, highly structured Markdown. Do NOT include markdown blocks of code (like \`\`\`markdown). Use bullet points and bold text where relevant to highlight operational thresholds (e.g. kW thresholds). Keep the tone technical, executive, and decisive. Let's make it look fantastic!`;

      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${geminiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }]
        })
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.error?.message || `HTTP ${response.status} Error`);
      }

      const resData = await response.json();
      const text = resData.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!text) {
        throw new Error('Gemini API returned an empty response.');
      }

      setAiReport(text);

    } catch (err) {
      console.error('[GEMINI] Fetch failed:', err);
      setAiError(err.message || 'Failed to generate AI advice. Please check your API key and connection.');
    } finally {
      setAiLoading(false);
    }
  }, [geminiKey, liveForecastResults, lat, lon, getMockAiReport]);



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

          {/* Sub-page Navigation Tabs */}
          <div className="header-tabs">
            <button
              className={`tab-btn ${activeTab === 'sandbox' ? 'active' : ''}`}
              onClick={() => setActiveTab('sandbox')}
            >
              📊 Sandbox Simulation
            </button>
            <button
              className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              📡 Live Forecast Hub
            </button>
          </div>
        </div>
        <div className="header-actions">
          {activeTab === 'sandbox' && (
            <div className="dataset-toggle">
              <button id="btn-uci" className={dataset === 'uci' ? 'active' : ''} onClick={() => setDataset('uci')}>
                UCI Solar
              </button>
              <button id="btn-aep" className={dataset === 'aep' ? 'active' : ''} onClick={() => setDataset('aep')}>
                AEP Grid
              </button>
            </div>
          )}
          <button className="btn-export" onClick={exportCSV} disabled={!results} id="btn-export">
            ⬇ Export CSV
          </button>
          {activeTab === 'sandbox' && (
            <button
              id="btn-run-model"
              className={`btn-run ${modelStatus === 'training' ? 'running' : ''}`}
              onClick={runModel}
              disabled={modelStatus === 'training'}
            >
              {modelStatus === 'training' ? '⚡ Running…' : modelStatus === 'done' ? '↺ Re-Run' : '▶ Run Model'}
            </button>
          )}
        </div>
      </header>

      {/* ---- Main ---- */}
      <main className="main">
        {activeTab === 'sandbox' ? (
          <>
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

            {/* ---- Results ---- */}
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
          </>
        ) : (
          <>
            {/* ============================================================
               Live Forecast Hub View
               ============================================================ */}

            {/* API and Location Configuration */}
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">
                  <div className="dot dot-orion" />
                  Live Forecast Hub Configuration
                </div>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Provide your coordinates and API keys to forecast solar production in real time
                </span>
              </div>

              <div className="settings-grid">
                <div className="settings-field">
                  <label>Weather API Key (OpenWeatherMap)</label>
                  <div className="settings-input-group">
                    <input
                      type={showWeatherKey ? 'text' : 'password'}
                      className="settings-input"
                      placeholder="Enter OpenWeatherMap API Key"
                      value={weatherKey}
                      onChange={e => setWeatherKey(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn-toggle-visibility"
                      onClick={() => setShowWeatherKey(!showWeatherKey)}
                    >
                      {showWeatherKey ? '👁️' : '🕶️'}
                    </button>
                  </div>
                </div>

                <div className="settings-field">
                  <label>Gemini API Key (Optional)</label>
                  <div className="settings-input-group">
                    <input
                      type={showGeminiKey ? 'text' : 'password'}
                      className="settings-input"
                      placeholder="Enter Google Gemini API Key"
                      value={geminiKey}
                      onChange={e => setGeminiKey(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn-toggle-visibility"
                      onClick={() => setShowGeminiKey(!showGeminiKey)}
                    >
                      {showGeminiKey ? '👁️' : '🕶️'}
                    </button>
                  </div>
                </div>

                <div className="settings-field">
                  <label>Latitude</label>
                  <input
                    type="text"
                    className="settings-input"
                    placeholder="e.g. 36.1716"
                    value={lat}
                    onChange={e => setLat(e.target.value)}
                  />
                </div>

                <div className="settings-field">
                  <label>Longitude</label>
                  <input
                    type="text"
                    className="settings-input"
                    placeholder="e.g. -115.1398"
                    value={lon}
                    onChange={e => setLon(e.target.value)}
                  />
                </div>

                <button
                  className="btn-fetch"
                  onClick={fetchLiveForecast}
                  disabled={weatherLoading}
                >
                  {weatherLoading ? '⚡ Fetching...' : '📡 Fetch Live Forecast'}
                </button>
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                💡 <strong>Don't have keys?</strong> You can get a free weather API key by signing up on <a href="https://openweathermap.org/price" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--orion)' }}>OpenWeatherMap</a> (the basic 5-Day Forecast is completely free). To try the Gemini AI Grid Advisor, you can leave the Gemini key blank to run an **interactive demo report** with high-fidelity mock data!
              </div>
            </div>

            {/* Error Message */}
            {weatherError && (
              <div className="error-message fade-in">
                <span>⚠️</span>
                <span>{weatherError}</span>
              </div>
            )}

            {/* Weather overview & current forecast cards */}
            {currentWeather && (
              <div className="weather-overview-grid fade-in">
                <div className="weather-widget">
                  <div className="weather-widget-label">Target Location</div>
                  <div className="weather-widget-val" style={{ fontSize: 13, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                    {weatherLocation}
                  </div>
                  <div className="weather-widget-sub">Lat: {lat} · Lon: {lon}</div>
                </div>
                <div className="weather-widget">
                  <div className="weather-widget-label">Temperature</div>
                  <div className="weather-widget-val">{currentWeather.tmp.toFixed(1)}°C</div>
                  <div className="weather-widget-sub">Forecasted Ambient</div>
                </div>
                <div className="weather-widget">
                  <div className="weather-widget-label">Cloud Cover</div>
                  <div className="weather-widget-val">{Math.round(currentWeather.cc * 100)}%</div>
                  <div className="weather-widget-sub">Atmospheric density</div>
                </div>
                <div className="weather-widget">
                  <div className="weather-widget-label">Est. Solar Irradiance</div>
                  <div className="weather-widget-val">{Math.round(currentWeather.irr)} W/m²</div>
                  <div className="weather-widget-sub">Normalised input proxy</div>
                </div>
                <div className="weather-widget">
                  <div className="weather-widget-label">Humidity & Wind</div>
                  <div className="weather-widget-val" style={{ fontSize: 13 }}>
                    💦 {Math.round(currentWeather.hum)}% · 💨 {currentWeather.wnd.toFixed(1)}m/s
                  </div>
                  <div className="weather-widget-sub">Relative moisture / speed</div>
                </div>
              </div>
            )}

            {/* Weather Prediction Line Chart */}
            {liveForecastResults && (
              <div className="card fade-in">
                <div className="card-header">
                  <div className="card-title">
                    <div className="dot dot-green" />
                    Live 5-Day Solar Power Forecast
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Sequentially forecasted using ORION (Echo State Recurrent) + AURORA (Adaptive Correction)
                  </div>
                </div>

                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={liveForecastResults.chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="gradLiveAurora" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={C.aurora} stopOpacity={0.15} />
                          <stop offset="95%" stopColor={C.aurora} stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradLiveIrr" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={C.gold} stopOpacity={0.08} />
                          <stop offset="95%" stopColor={C.gold} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(224,20%,14%)" />
                      <XAxis dataKey="label" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `${v}kW`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="aurora" name="AURORA Refined" stroke={C.aurora} strokeWidth={2.5} fill="url(#gradLiveAurora)" dot={false} />
                      <Line type="monotone" dataKey="orion" name="ORION Initial" stroke={C.orion} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                      <Area type="monotone" dataKey="irr" name="Solar Irradiance" stroke={C.gold} strokeWidth={1} fill="url(#gradLiveIrr)" dot={false} opacity={0.4} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-legend">
                  {[
                    { color: C.aurora, label: 'AURORA Final Solar Power Forecast' },
                    { color: C.orion, label: 'ORION Initial Forecast (Raw Reservoir)' },
                    { color: C.gold, label: 'Estimated Irradiance (W/m²)' },
                  ].map(({ color, label }) => (
                    <div className="legend-item" key={label}>
                      <div className="legend-dot" style={{ background: color }} />
                      {label}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Gemini AI Advisor & Forecast Details Table */}
            {liveForecastResults && (
              <div className="grid-2">
                {/* Gemini AI advisor card */}
                <div className="card card-ai fade-in">
                  <div className="ai-header">
                    <div className="ai-badge">🤖 Gemini AI Grid Advisor</div>
                    <button
                      className="ai-btn"
                      onClick={generateAiAdvice}
                      disabled={aiLoading}
                    >
                      {aiLoading ? '⚡ Consulting AI...' : '✨ Generate AI Advice'}
                    </button>
                  </div>

                  {aiError && (
                    <div className="error-message">
                      <span>⚠️</span>
                      <span>{aiError}</span>
                    </div>
                  )}

                  {aiLoading ? (
                    <div className="skeleton-wrap">
                      <div className="skeleton-bar" style={{ width: '40%' }} />
                      <div className="skeleton-bar" style={{ width: '90%' }} />
                      <div className="skeleton-bar" style={{ width: '85%' }} />
                      <div className="skeleton-bar" style={{ width: '70%' }} />
                      <div className="skeleton-bar" style={{ width: '80%' }} />
                      <div className="skeleton-bar" style={{ width: '50%' }} />
                    </div>
                  ) : aiReport ? (
                    <div className="ai-content">
                      {!geminiKey && (
                        <div style={{
                          padding: '8px 12px', background: 'rgba(251, 191, 36, 0.08)',
                          border: '1px solid rgba(251, 191, 36, 0.25)', borderRadius: 8,
                          fontSize: 11, color: 'var(--gold)', marginBottom: 16
                        }}>
                          💡 <strong>Viewing Interactive Demo Report</strong>. Provide a Gemini API Key in the settings bar to get live customized grid dispatch summaries!
                        </div>
                      )}
                      
                      {/* Simple custom markdown renderer using pure JS */}
                      {aiReport.split('\n').map((line, idx) => {
                        const trimmed = line.trim();
                        if (trimmed.startsWith('###')) {
                          return <h3 key={idx}>{trimmed.replace('###', '').trim()}</h3>;
                        } else if (trimmed.startsWith('-') || trimmed.startsWith('*')) {
                          // bold processing
                          const parts = trimmed.substring(1).split('**');
                          return (
                            <li key={idx} style={{ marginLeft: 16, listStyleType: 'disc', margin: '4px 0 4px 16px' }}>
                              {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx}>{part}</strong> : part)}
                            </li>
                          );
                        } else if (trimmed === '') {
                          return <div key={idx} style={{ height: 8 }} />;
                        } else {
                          const parts = trimmed.split('**');
                          return (
                            <p key={idx}>
                              {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx}>{part}</strong> : part)}
                            </p>
                          );
                        }
                      })}
                    </div>
                  ) : (
                    <div style={{ padding: '32px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
                      <div style={{ fontSize: 32, marginBottom: 12 }}>🤖</div>
                      <p>Predict solar output above, then click <strong>Generate AI Advice</strong> to get high-value battery energy storage scheduling, grid load-shifting recommendations, and smart-home operational planning!</p>
                    </div>
                  )}
                </div>

                {/* Weather details scrolling list */}
                <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column' }}>
                  <div className="card-header">
                    <div className="card-title">
                      <div className="dot dot-orion" />
                      Forecast Generation Details
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>48-hour sequential time slots</span>
                  </div>

                  <div style={{ flex: 1, overflowY: 'auto', maxHeight: 380, paddingRight: 4 }} className="scroll-container">
                    <table className="bench-table" style={{ width: '100%' }}>
                      <thead>
                        <tr>
                          <th>Time</th>
                          <th>Temp</th>
                          <th>Clouds</th>
                          <th>Irradiance</th>
                          <th>ORION</th>
                          <th>AURORA</th>
                        </tr>
                      </thead>
                      <tbody>
                        {liveForecastResults.chartData.map((row, i) => (
                          <tr key={i} className={i % 2 === 0 ? 'highlight' : ''}>
                            <td className="model-name-cell" style={{ fontSize: 11 }}>
                              {row.dateLabel} {row.label}
                            </td>
                            <td>{row.temp.toFixed(1)}°C</td>
                            <td>{row.cloud}%</td>
                            <td style={{ color: 'var(--gold)' }}>{row.irr} W/m²</td>
                            <td style={{ color: 'var(--orion)' }}>{row.orion} kW</td>
                            <td style={{ color: 'var(--aurora)', fontWeight: 700 }}>{row.aurora} kW</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Setup prompt card if weather forecast is not loaded yet */}
            {!currentWeather && (
              <div className="card fade-in" style={{ textAlign: 'center', padding: '48px 24px' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>📡</div>
                <h2 className="text-gradient" style={{ fontSize: 24, fontWeight: 800, marginBottom: 12 }}>
                  Live Solar Forecasting Hub
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: 13, maxWidth: 520, margin: '0 auto 20px', lineHeight: 1.6 }}>
                  Enter your weather API key and coordinates in the configuration panel above to calculate a real-time 5-day solar power forecast. The system uses your live local weather parameters to query the Echo State Network dynamically!
                </p>
              </div>
            )}
          </>
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
