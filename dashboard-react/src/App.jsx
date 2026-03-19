import { useState, useMemo } from "react";
import { usePageData } from "./hooks/usePageData";

// ── Edge function URLs ────────────────────────────────────────────────────────
const BASE = import.meta.env.VITE_EDGE_BASE_URL || "";
const URLS = {
  core:        `${BASE}/dashboard-core`,
  predictions: `${BASE}/dashboard-predictions`,
  simulation:  `${BASE}/dashboard-simulation`,
  market:      `${BASE}/dashboard-market`,
};

// ── Tiny helpers ──────────────────────────────────────────────────────────────
const pct  = (v, d = 1) => v == null ? "—" : `${(v * 100).toFixed(d)}%`;
const fmt2 = (v) => v == null ? "—" : Number(v).toFixed(2);
const fmt4 = (v) => v == null ? "—" : Number(v).toFixed(4);
const fmtK = (v) => v == null ? "—" : v >= 1e9 ? `$${(v/1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v/1e6).toFixed(1)}M` : `$${Number(v).toLocaleString()}`;
// For metrics already stored as percentage numbers (e.g. total_return_pct = 15.3 means 15.3%)
const fmtPct = (v, d = 2) => v == null ? "—" : `${Number(v).toFixed(d)}%`;
const clsNum = (v) => v == null ? "" : v >= 0 ? "pos" : "neg";
const ts = (iso) => { if (!iso) return "—"; const d = new Date(iso); return d.toLocaleString(); };
const dateOnly = (s) => s ? String(s).slice(0, 10) : "—";

// ── Inline SVG charts ─────────────────────────────────────────────────────────
function Sparkline({ values = [], color = "auto", height = 32, width = 100, fill = true, dots = false }) {
  if (!values || values.length < 2) return <span className="no-data">—</span>;
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const last = values[values.length - 1];
  const first = values[0];
  const autoColor = last >= first ? "#22c98b" : "#f05c6a";
  const c = color === "auto" ? autoColor : color;
  const polyPts = pts.join(" ");
  const fillPts = `0,${height} ${polyPts} ${width},${height}`;
  const gid = `sg${c.replace("#", "")}`;
  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={c} stopOpacity="0.3" />
          <stop offset="100%" stopColor={c} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {fill && <polygon points={fillPts} fill={`url(#${gid})`} />}
      <polyline points={polyPts} fill="none" stroke={c} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      {dots && <circle cx={pts[pts.length - 1].split(",")[0]} cy={pts[pts.length - 1].split(",")[1]} r="2.5" fill={c} />}
    </svg>
  );
}

function IcBarChart({ series = [], height = 48, width = 200 }) {
  if (!series || series.length === 0) return <span className="no-data">No IC data</span>;
  const vals = series.map((s) => s.ic ?? s);
  const maxAbs = Math.max(...vals.map(Math.abs), 0.001);
  const barW = Math.max(2, width / vals.length - 1);
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="#ffffff10" strokeWidth="1" />
      {vals.map((v, i) => {
        const barH = Math.max(1, Math.abs(v) / maxAbs * (height / 2 - 2));
        const x = i * (barW + 1);
        const y = v >= 0 ? height / 2 - barH : height / 2;
        return <rect key={i} x={x} y={y} width={barW} height={barH} fill={v >= 0 ? "#22c98b" : "#f05c6a"} rx="1" />;
      })}
    </svg>
  );
}

function PnlBar({ value, max = 1000 }) {
  const p = Math.min(Math.abs(value ?? 0) / (max || 1), 1) * 100;
  const c = (value ?? 0) >= 0 ? "#22c98b" : "#f05c6a";
  return (
    <div className="pnl-bar-wrap">
      <div className="pnl-bar-track"><div className="pnl-bar-fill" style={{ width: `${p}%`, background: c }} /></div>
      <span className={`pnl-val ${(value ?? 0) >= 0 ? "pos" : "neg"}`}>{(value ?? 0) >= 0 ? "+" : ""}{fmt2(value)}</span>
    </div>
  );
}

function StatusDot({ status, pulse = false }) {
  const cls = status === "healthy" || status === "success" || status === "completed" ? "ok"
    : status === "degraded" || status === "warning" ? "warn" : "bad";
  return <span className={`dot dot-${cls}${pulse ? " pulse" : ""}`} />;
}

function Badge({ text, variant = "neutral" }) {
  return <span className={`badge badge-${variant}`}>{text}</span>;
}

function Stat({ label, value, sub, color }) {
  return (
    <div className="stat-cell">
      <div className="stat-val" style={color ? { color } : {}}>{value ?? "—"}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function Card({ title, children, accent, className = "" }) {
  return (
    <div className={`card ${className}`} style={accent ? { borderTop: `2px solid ${accent}` } : {}}>
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}

function Spinner() { return <div className="spinner" />; }
function ErrorBanner({ msg }) { return <div className="error-banner">{msg}</div>; }
function EmptyState({ msg = "No data available" }) { return <div className="empty-state">{msg}</div>; }

// ── NAV CONFIG ────────────────────────────────────────────────────────────────
const NAV = [
  { id: "overview",    label: "Overview",      icon: "⬡",  group: "main" },
  { id: "predictions", label: "Predictions",   icon: "◈",  group: "main" },
  { id: "signal",      label: "Signal Health", icon: "∿",  group: "main" },
  { id: "simulation",  label: "Simulation",    icon: "⟳",  group: "quant" },
  { id: "trades",      label: "Trades",        icon: "⇅",  group: "quant" },
  { id: "market",      label: "Market",        icon: "◉",  group: "quant" },
  { id: "pipeline",    label: "Pipeline",      icon: "⎇",  group: "system" },
  { id: "model",       label: "Model",         icon: "⬙",  group: "system" },
];

// ── PAGES ─────────────────────────────────────────────────────────────────────

function OverviewPage({ coreData, simData, predData, mktData }) {
  const mon    = coreData?.monitoring ?? {};
  const sim    = simData?.latest_run_metrics ?? {};
  const aq     = predData?.alpha_quality ?? {};
  const sh     = predData?.signal_health ?? {};
  const ctx    = mktData?.market_context ?? {};
  const alerts = mon.alert_details ?? [];

  return (
    <div className="page-content">
      <div className="kpi-strip">
        <Stat label="Active Symbols"    value={mon.data_quality?.active_symbols ?? "—"} />
        <Stat label="Missing Bars 30d"  value={mon.data_quality?.missing_pct != null ? `${mon.data_quality.missing_pct}%` : "—"}
          color={mon.data_quality?.missing_pct > 1 ? "#f05c6a" : "#22c98b"} />
        <Stat label="Quarantine 7d"     value={mon.data_quality?.quarantine_7d ?? "—"}
          color={(mon.data_quality?.quarantine_7d ?? 0) > 3 ? "#f0a843" : undefined} />
        <Stat label="IC Mean"           value={fmt4(sh.ic_mean)} sub="all history"
          color={sh.ic_mean > 0 ? "#22c98b" : sh.ic_mean < 0 ? "#f05c6a" : undefined} />
        <Stat label="IC 30d"            value={fmt4(sh.ic_30d)} sub={sh.ic_trend}
          color={sh.ic_trend === "improving" ? "#22c98b" : sh.ic_trend === "declining" ? "#f05c6a" : undefined} />
        <Stat label="VIX"               value={ctx.vix ?? "—"} sub={ctx.vix_label}
          color={ctx.vix >= 30 ? "#f05c6a" : ctx.vix >= 20 ? "#f0a843" : "#22c98b"} />
        <Stat label="Sharpe (sim)"      value={fmt2(sim.sharpe_ratio)} />
        <Stat label="Hit Rate 60–70%"   value={aq.hit_rates?.["60–70%"]?.hit_rate != null ? pct(aq.hit_rates["60–70%"].hit_rate) : "—"} />
      </div>

      <div className="overview-row">
        <Card title="Alerts" accent={alerts.length > 0 ? "#f05c6a" : "#22c98b"}>
          {alerts.length === 0
            ? <div className="alert-ok">All systems healthy</div>
            : alerts.map((a, i) => (
              <div key={i} className={`alert-row alert-${a.level}`}>
                <span className="alert-level">{a.level.toUpperCase()}</span>
                <span className="alert-check">{a.check}</span>
                <span className="alert-detail">{a.detail}</span>
              </div>
            ))}
        </Card>

        <Card title="Data Freshness">
          <table className="info-table">
            <tbody>
              <tr><td>Latest Market Bar</td>   <td>{dateOnly(mon.data_freshness?.latest_market_bar)}</td></tr>
              <tr><td>Latest Feature</td>       <td>{dateOnly(mon.data_freshness?.latest_feature)}</td></tr>
              <tr><td>Latest Prediction</td>    <td>{ts(mon.data_freshness?.latest_prediction)}</td></tr>
              <tr><td>Bar Days 7d</td>          <td>{mon.pipeline_health?.market_bar_days_7d ?? "—"}</td></tr>
              <tr><td>Feature Days 7d</td>      <td>{mon.pipeline_health?.feature_days_7d ?? "—"}</td></tr>
              <tr><td>Prediction Days 7d</td>   <td>{mon.pipeline_health?.prediction_days_7d ?? "—"}</td></tr>
            </tbody>
          </table>
        </Card>

        <Card title="Market">
          <table className="info-table">
            <tbody>
              <tr><td>VIX</td>           <td style={{ color: ctx.vix >= 30 ? "#f05c6a" : ctx.vix >= 20 ? "#f0a843" : "#22c98b" }}>{ctx.vix ?? "—"} ({ctx.vix_label ?? "—"})</td></tr>
              <tr><td>Regime</td>        <td>{ctx.regime ?? "—"}</td></tr>
              <tr><td>SPY 1d</td>        <td className={clsNum(ctx.spy_1d_return_pct)}>{ctx.spy_1d_return_pct != null ? `${ctx.spy_1d_return_pct > 0 ? "+" : ""}${ctx.spy_1d_return_pct}%` : "—"}</td></tr>
              <tr><td>SPY Close</td>     <td>{ctx.spy_last_close != null ? `$${ctx.spy_last_close}` : "—"}</td></tr>
              <tr><td>60d Drawdown</td>  <td className="neg">{ctx.spy_drawdown_60d != null ? pct(ctx.spy_drawdown_60d) : "—"}</td></tr>
              <tr><td>SP500 Mom 200d</td><td className={clsNum(ctx.sp500_momentum_200d)}>{ctx.sp500_momentum_200d != null ? `${ctx.sp500_momentum_200d > 0 ? "+" : ""}${ctx.sp500_momentum_200d}%` : "—"}</td></tr>
            </tbody>
          </table>
          {mktData?.spy_ohlcv?.length > 1 && (
            <div style={{ marginTop: "8px" }}>
              <Sparkline values={mktData.spy_ohlcv.map((r) => r.close)} width={220} height={40} color="auto" fill />
            </div>
          )}
        </Card>
      </div>

      {Object.keys(sim).length > 0 && (
        <Card title="Latest Simulation">
          <div className="kpi-strip kpi-strip-sm">
            <Stat label="Ann. Return"  value={fmtPct(sim.annualized_return_pct)} />
            <Stat label="Sharpe"       value={fmt2(sim.sharpe_ratio)} />
            <Stat label="Sharpe NW"    value={fmt2(sim.sharpe_ratio_nw)} />
            <Stat label="Max DD"       value={fmtPct(sim.max_drawdown_pct)} color="#f05c6a" />
            <Stat label="Win Rate"     value={sim.win_rate != null ? pct(sim.win_rate) : "—"} />
            <Stat label="IC Mean"      value={fmt4(sim.ic_mean)} />
            <Stat label="N Trades"     value={sim.total_trades ?? "—"} />
            <Stat label="Net Profit"   value={sim.net_profit != null ? `$${fmt2(sim.net_profit)}` : "—"} />
          </div>
        </Card>
      )}
    </div>
  );
}

function PredictionsPage({ data }) {
  const [search, setSearch]   = useState("");
  const [dirFilter, setDir]   = useState("all");
  const [minConf, setMinConf] = useState(0);
  const [withOutcome, setWO]  = useState(false);
  const [sortCol, setSortCol] = useState("target_date");
  const [sortAsc, setSortAsc] = useState(false);

  const preds = data?.predictions ?? [];
  const filtered = useMemo(() => {
    let rows = preds;
    if (search)      rows = rows.filter((r) => r.symbol?.toLowerCase().includes(search.toLowerCase()));
    if (dirFilter !== "all") rows = rows.filter((r) => r.direction === dirFilter);
    if (minConf > 0) rows = rows.filter((r) => (r.confidence ?? r.probability_up ?? 0) >= minConf);
    if (withOutcome) rows = rows.filter((r) => r.realized_direction != null);
    return [...rows].sort((a, b) => {
      const va = a[sortCol] ?? 0, vb = b[sortCol] ?? 0;
      return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    });
  }, [preds, search, dirFilter, minConf, withOutcome, sortCol, sortAsc]);

  function TH({ col, label }) {
    return (
      <th onClick={() => { setSortAsc(sortCol === col ? !sortAsc : false); setSortCol(col); }}
          style={{ cursor: "pointer", userSelect: "none" }}>
        {label}{sortCol === col ? (sortAsc ? " ↑" : " ↓") : ""}
      </th>
    );
  }

  return (
    <div className="page-content">
      <div className="filter-bar">
        <input className="filter-input" placeholder="Symbol…" value={search} onChange={(e) => setSearch(e.target.value)} />
        <select className="filter-select" value={dirFilter} onChange={(e) => setDir(e.target.value)}>
          <option value="all">All directions</option>
          <option value="up">Up</option>
          <option value="down">Down</option>
          <option value="outperform">Outperform</option>
          <option value="underperform">Underperform</option>
        </select>
        <select className="filter-select" value={minConf} onChange={(e) => setMinConf(Number(e.target.value))}>
          <option value={0}>All confidence</option>
          <option value={0.55}>≥ 55%</option>
          <option value={0.60}>≥ 60%</option>
          <option value={0.70}>≥ 70%</option>
        </select>
        <label className="filter-check">
          <input type="checkbox" checked={withOutcome} onChange={(e) => setWO(e.target.checked)} />
          With outcome
        </label>
        <span className="filter-count">{filtered.length} rows</span>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <TH col="symbol"       label="Symbol" />
              <TH col="target_date"  label="Date" />
              <TH col="direction"    label="Direction" />
              <TH col="probability_up" label="P(Up)" />
              <TH col="confidence"   label="Confidence" />
              <TH col="model_version" label="Model" />
              <th>Outcome</th>
              <TH col="realized_return" label="Return" />
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 500).map((r) => {
              const conf    = r.confidence ?? r.probability_up ?? 0;
              const correct = r.realized_direction && r.direction === r.realized_direction;
              return (
                <tr key={r.prediction_id}>
                  <td className="sym">{r.symbol}</td>
                  <td>{dateOnly(r.target_date)}</td>
                  <td><span className={`dir-badge dir-${r.direction}`}>{r.direction ?? "—"}</span></td>
                  <td>{r.probability_up != null ? pct(r.probability_up) : "—"}</td>
                  <td>
                    <div className="conf-bar-wrap">
                      <div className="conf-bar" style={{ width: `${(conf*100).toFixed(0)}%`,
                        background: conf >= 0.7 ? "#22c98b" : conf >= 0.6 ? "#3b7cf8" : "#f0a843" }} />
                      <span>{pct(conf)}</span>
                    </div>
                  </td>
                  <td className="mono">{r.model_version ?? "—"}</td>
                  <td>{r.realized_direction
                    ? <span className={correct ? "outcome-ok" : "outcome-bad"}>{correct ? "✓" : "✗"} {r.realized_direction}</span>
                    : <span className="muted">pending</span>}
                  </td>
                  <td className={clsNum(r.realized_return)}>
                    {r.realized_return != null ? `${(r.realized_return * 100).toFixed(2)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SignalPage({ data }) {
  const sh     = data?.signal_health ?? {};
  const aq     = data?.alpha_quality ?? {};
  const series = sh.ic_series ?? [];

  return (
    <div className="page-content">
      <div className="kpi-strip">
        <Stat label="IC Mean"      value={fmt4(sh.ic_mean)}
          color={sh.ic_mean > 0 ? "#22c98b" : sh.ic_mean < 0 ? "#f05c6a" : undefined} />
        <Stat label="IC 30d"       value={fmt4(sh.ic_30d)} />
        <Stat label="IC Prior 30d" value={fmt4(sh.ic_prior_30d)} />
        <Stat label="Trend"        value={sh.ic_trend ?? "—"}
          color={sh.ic_trend === "improving" ? "#22c98b" : sh.ic_trend === "declining" ? "#f05c6a" : "#f0a843"} />
        <Stat label="Decile Spread" value={sh.decile_spread_bps != null ? `${sh.decile_spread_bps} bps` : "—"} />
        <Stat label="IC Days"      value={sh.n_ic_days ?? "—"} />
        <Stat label="Outcomes 30d" value={aq.n_outcomes_30d ?? "—"} />
        <Stat label="Outcomes 120d" value={aq.n_outcomes_120d ?? "—"} />
      </div>

      <div className="overview-row">
        <Card title={`Daily IC — last ${series.length} days`} className="card-wide">
          <div className="ic-chart-wrap">
            <IcBarChart series={series} height={90} width={Math.min(series.length * 8, 640)} />
          </div>
          <div className="muted" style={{ marginTop: "6px", fontSize: "0.72rem" }}>
            Each bar = cross-sectional Spearman IC for that trading day · green = positive
          </div>
        </Card>

        <Card title="Hit Rates by Confidence">
          <table className="data-table">
            <thead><tr><th>Tier</th><th>N</th><th>Hit Rate</th><th>Edge</th></tr></thead>
            <tbody>
              {Object.entries(aq.hit_rates ?? {}).map(([tier, v]) => (
                <tr key={tier}>
                  <td>{tier}</td>
                  <td>{v.n}</td>
                  <td className={v.hit_rate > 0.5 ? "pos" : "neg"}>{pct(v.hit_rate)}</td>
                  <td className={v.hit_rate > 0.5 ? "pos" : "neg"}>{v.hit_rate > 0.5 ? "+" : ""}{pct(v.hit_rate - 0.5)}</td>
                </tr>
              ))}
              {Object.keys(aq.hit_rates ?? {}).length === 0 && (
                <tr><td colSpan={4} className="muted">Awaiting outcomes</td></tr>
              )}
            </tbody>
          </table>
        </Card>
      </div>

      {series.length > 1 && (
        <Card title="Decile Spread (bps) — rolling">
          <Sparkline values={series.map((s) => s.spread)} width={600} height={60} color="auto" fill />
        </Card>
      )}
    </div>
  );
}

function SimulationPage({ data }) {
  const runs = data?.runs ?? [];

  return (
    <div className="page-content">
      {runs.length === 0 && <EmptyState msg="No completed simulation runs found." />}

      {runs[0] && (
        <div className="kpi-strip">
          <Stat label="Total Return"  value={fmtPct(runs[0].metrics?.total_return_pct)} />
          <Stat label="Ann. Return"   value={fmtPct(runs[0].metrics?.annualized_return_pct)} />
          <Stat label="Sharpe"        value={fmt2(runs[0].metrics?.sharpe_ratio)} />
          <Stat label="Sharpe NW"     value={fmt2(runs[0].metrics?.sharpe_ratio_nw)} />
          <Stat label="Max DD"        value={fmtPct(runs[0].metrics?.max_drawdown_pct)} color="#f05c6a" />
          <Stat label="Win Rate"      value={runs[0].metrics?.win_rate != null ? pct(runs[0].metrics.win_rate) : "—"} />
          <Stat label="IC Mean"       value={fmt4(runs[0].metrics?.ic_mean)} />
          <Stat label="N Trades"      value={runs[0].metrics?.total_trades ?? "—"} />
        </div>
      )}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Run ID</th><th>Strategy</th><th>Model</th><th>Period</th>
              <th>Capital</th><th>Return</th><th>Ann. Return</th><th>Sharpe</th>
              <th>Max DD</th><th>Win Rate</th><th>IC Mean</th><th>Trades</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td className="mono" style={{ fontSize: "0.7rem" }}>{r.run_id.slice(0, 8)}</td>
                <td>{r.strategy_name}</td>
                <td className="mono">{r.model_version ?? "—"}</td>
                <td style={{ fontSize: "0.75rem" }}>{dateOnly(r.start_date)}→{dateOnly(r.end_date)}</td>
                <td>{fmtK(r.starting_capital)}</td>
                <td className={clsNum(r.metrics?.total_return_pct)}>{fmtPct(r.metrics?.total_return_pct)}</td>
                <td className={clsNum(r.metrics?.annualized_return_pct)}>{fmtPct(r.metrics?.annualized_return_pct)}</td>
                <td className={r.metrics?.sharpe_ratio > 1 ? "pos" : "neg"}>{fmt2(r.metrics?.sharpe_ratio)}</td>
                <td className="neg">{fmtPct(r.metrics?.max_drawdown_pct)}</td>
                <td>{r.metrics?.win_rate != null ? pct(r.metrics.win_rate) : "—"}</td>
                <td>{fmt4(r.metrics?.ic_mean)}</td>
                <td>{r.metrics?.total_trades ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TradesPage({ data }) {
  const trades = data?.latest_trades ?? [];
  const stats  = data?.trade_stats ?? {};
  const maxPnl = Math.max(...trades.map((t) => Math.abs(t.daily_pnl ?? 0)), 1);

  return (
    <div className="page-content">
      <div className="kpi-strip">
        <Stat label="Total P&L" value={`$${fmt2(stats.total_pnl)}`} color={(stats.total_pnl ?? 0) >= 0 ? "#22c98b" : "#f05c6a"} />
        <Stat label="N Trades"  value={stats.n_trades ?? "—"} />
        <Stat label="Winners"   value={stats.n_winners ?? "—"} color="#22c98b" />
        <Stat label="Losers"    value={stats.n_losers ?? "—"} color="#f05c6a" />
        <Stat label="Win Rate"  value={stats.win_rate != null ? pct(stats.win_rate) : "—"}
          color={(stats.win_rate ?? 0) > 0.5 ? "#22c98b" : "#f05c6a"} />
      </div>

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr><th>Date</th><th>Symbol</th><th>Weight</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Tx Cost</th><th>Slip</th><th>Run</th></tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={i}>
                <td>{dateOnly(t.date)}</td>
                <td className="sym">{t.symbol}</td>
                <td>{t.weight != null ? pct(t.weight, 2) : "—"}</td>
                <td>{t.position_qty != null ? Number(t.position_qty).toFixed(0) : "—"}</td>
                <td>{fmt2(t.entry_price)}</td>
                <td>{fmt2(t.exit_price)}</td>
                <td><PnlBar value={t.daily_pnl} max={maxPnl} /></td>
                <td className="muted">{t.transaction_cost != null ? `$${fmt2(t.transaction_cost)}` : "—"}</td>
                <td className="muted">{t.slippage_cost != null ? `$${fmt2(t.slippage_cost)}` : "—"}</td>
                <td className="mono" style={{ fontSize: "0.7rem" }}>{t.run_id?.slice(0, 8) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MarketPage({ data }) {
  const ctx   = data?.market_context ?? {};
  const ohlcv = data?.spy_ohlcv ?? [];
  const closes  = ohlcv.map((r) => r.close);
  const volumes = ohlcv.map((r) => r.volume ?? 0);
  const vixColor = ctx.vix >= 40 ? "#f05c6a" : ctx.vix >= 30 ? "#f0a843" : ctx.vix >= 20 ? "#f0a843" : "#22c98b";

  return (
    <div className="page-content">
      <div className="kpi-strip">
        <Stat label="VIX"            value={ctx.vix ?? "—"} sub={ctx.vix_label} color={vixColor} />
        <Stat label="Regime"         value={ctx.regime ?? "—"} />
        <Stat label="SPY 1d Return"  value={ctx.spy_1d_return_pct != null ? `${ctx.spy_1d_return_pct > 0 ? "+" : ""}${ctx.spy_1d_return_pct}%` : "—"}
          color={(ctx.spy_1d_return_pct ?? 0) >= 0 ? "#22c98b" : "#f05c6a"} />
        <Stat label="SPY Last Close" value={ctx.spy_last_close != null ? `$${ctx.spy_last_close}` : "—"} />
        <Stat label="60d Drawdown"   value={ctx.spy_drawdown_60d != null ? pct(ctx.spy_drawdown_60d) : "—"} color="#f05c6a" />
        <Stat label="SP500 Mom 200d" value={ctx.sp500_momentum_200d != null ? `${ctx.sp500_momentum_200d > 0 ? "+" : ""}${ctx.sp500_momentum_200d}%` : "—"}
          color={(ctx.sp500_momentum_200d ?? 0) >= 0 ? "#22c98b" : "#f05c6a"} />
        <Stat label="52w High"       value={ctx.spy_52w_high != null ? `$${ctx.spy_52w_high}` : "—"} />
        <Stat label="From 52w High"  value={ctx.pct_from_52w_high != null ? `${ctx.pct_from_52w_high}%` : "—"}
          color={(ctx.pct_from_52w_high ?? 0) >= 0 ? "#22c98b" : "#f05c6a"} />
      </div>

      <div className="overview-row">
        {closes.length > 1 && (
          <Card title="SPY — 60-day Close" className="card-wide">
            <Sparkline values={closes} width={560} height={80} color="auto" fill dots />
            <div className="chart-x-labels">
              <span>{dateOnly(ohlcv[0]?.date)}</span>
              <span>{dateOnly(ohlcv[ohlcv.length - 1]?.date)}</span>
            </div>
          </Card>
        )}
        {volumes.length > 1 && (
          <Card title="SPY — Volume">
            <Sparkline values={volumes} width={280} height={60} color="#3b7cf8" fill />
          </Card>
        )}
      </div>

      <div className="table-wrap">
        <table className="data-table">
          <thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Δ%</th></tr></thead>
          <tbody>
            {[...ohlcv].reverse().slice(0, 30).map((r, i, arr) => {
              const prev = arr[i + 1];
              const chg  = prev ? ((r.close - prev.close) / prev.close * 100) : null;
              return (
                <tr key={r.date}>
                  <td>{dateOnly(r.date)}</td>
                  <td>{fmt2(r.open)}</td>
                  <td>{fmt2(r.high)}</td>
                  <td>{fmt2(r.low)}</td>
                  <td><b>{fmt2(r.close)}</b></td>
                  <td className="muted">{r.volume != null ? (r.volume / 1e6).toFixed(1) + "M" : "—"}</td>
                  <td className={chg != null ? clsNum(chg) : ""}>{chg != null ? `${chg > 0 ? "+" : ""}${chg.toFixed(2)}%` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PipelinePage({ data }) {
  const mon    = data?.monitoring ?? {};
  const wf     = data?.workflows ?? {};
  const alerts = mon.alert_details ?? [];

  const conclusionColor = (c) =>
    c === "success" ? "#22c98b" : c === "failure" ? "#f05c6a" : c === "in_progress" ? "#f0a843" : "#888";

  return (
    <div className="page-content">
      <div className="overview-row">
        <Card title="Alerts" accent={alerts.length > 0 ? "#f05c6a" : "#22c98b"}>
          {alerts.length === 0
            ? <div className="alert-ok">No alerts — all checks passing</div>
            : alerts.map((a, i) => (
              <div key={i} className={`alert-row alert-${a.level}`}>
                <span className="alert-level">{a.level.toUpperCase()}</span>
                <span className="alert-check">{a.check}</span>
                <span className="alert-detail">{a.detail}</span>
              </div>
            ))}
        </Card>

        <Card title="Pipeline Health">
          <table className="info-table">
            <tbody>
              <tr><td>Overall Status</td>    <td><StatusDot status={mon.overall_status} /> {mon.overall_status ?? "—"}</td></tr>
              <tr><td>Bar Days 7d</td>       <td>{mon.pipeline_health?.market_bar_days_7d ?? "—"}</td></tr>
              <tr><td>Feature Days 7d</td>   <td>{mon.pipeline_health?.feature_days_7d ?? "—"}</td></tr>
              <tr><td>Prediction Days 7d</td><td>{mon.pipeline_health?.prediction_days_7d ?? "—"}</td></tr>
              <tr><td>Active Symbols</td>    <td>{mon.data_quality?.active_symbols ?? "—"}</td></tr>
              <tr><td>Missing Bars 30d</td>  <td>{mon.data_quality?.missing_pct ?? "—"}%</td></tr>
              <tr><td>Quarantine 7d</td>     <td>{mon.data_quality?.quarantine_7d ?? "—"}</td></tr>
              <tr><td>Latest Bar</td>        <td>{dateOnly(mon.data_freshness?.latest_market_bar)}</td></tr>
              <tr><td>Latest Feature</td>    <td>{dateOnly(mon.data_freshness?.latest_feature)}</td></tr>
              <tr><td>Latest Prediction</td> <td>{ts(mon.data_freshness?.latest_prediction)}</td></tr>
            </tbody>
          </table>
        </Card>

        <Card title="GitHub Workflows">
          {!wf.available
            ? <div className="muted">{wf.reason ?? "Workflows unavailable"}</div>
            : (wf.runs ?? []).map((r) => (
              <div key={r.workflow} className="workflow-row">
                <StatusDot status={r.conclusion} />
                <div className="workflow-info">
                  <div className="workflow-name">{r.workflow.split("/").pop()?.replace(".yml", "")}</div>
                  <div className="workflow-meta">
                    <span style={{ color: conclusionColor(r.conclusion) }}>{r.conclusion}</span>
                    {r.run_number && <span className="muted"> · #{r.run_number}</span>}
                    {r.created_at && <span className="muted"> · {dateOnly(r.created_at)}</span>}
                  </div>
                </div>
                {r.html_url && <a href={r.html_url} target="_blank" rel="noreferrer" className="wf-link">↗</a>}
              </div>
            ))}
        </Card>
      </div>
    </div>
  );
}

function ModelPage({ data }) {
  const model = data?.model;
  if (!model) return <EmptyState msg="No production model found." />;
  const m = model.metrics ?? {};

  return (
    <div className="page-content">
      <div className="overview-row">
        <Card title="Model Registry" accent="#3b7cf8">
          <table className="info-table">
            <tbody>
              <tr><td>Version</td>          <td className="mono"><b>{model.model_version}</b></td></tr>
              <tr><td>Algorithm</td>        <td>{model.algorithm ?? "—"}</td></tr>
              <tr><td>Status</td>           <td><Badge text={model.status} variant={model.status === "production" ? "ok" : "warn"} /></td></tr>
              <tr><td>Training Window</td>  <td>{dateOnly(model.training_window_start)} → {dateOnly(model.training_window_end)}</td></tr>
              <tr><td>Created At</td>       <td>{ts(model.created_at)}</td></tr>
              <tr><td>Promoted At</td>      <td>{ts(model.promoted_at)}</td></tr>
            </tbody>
          </table>
        </Card>

        {Object.keys(m).length > 0 && (
          <Card title="Training Metrics">
            <div className="kpi-strip kpi-strip-sm">
              {m.val_ic              != null && <Stat label="Val IC"        value={fmt4(m.val_ic)} />}
              {m.seed_ic_mean        != null && <Stat label="Seed IC Mean"  value={fmt4(m.seed_ic_mean)} />}
              {m.auc_roc             != null && <Stat label="AUC-ROC"       value={fmt4(m.auc_roc)} />}
              {m.wf_sharpe_net       != null && <Stat label="Sharpe NW (WF)" value={fmt2(m.wf_sharpe_net)} />}
              {m.wf_deflated_sharpe  != null && <Stat label="DSR (WF)"      value={fmt2(m.wf_deflated_sharpe)} />}
              {m.wf_max_drawdown     != null && <Stat label="Max DD (WF)"   value={pct(m.wf_max_drawdown)} color="#f05c6a" />}
              {m.wf_ic_mean          != null && <Stat label="WF IC Mean"    value={fmt4(m.wf_ic_mean)} />}
              {m.oos_ic              != null && <Stat label="OOS IC"        value={fmt4(m.oos_ic)} />}
              {m.n_features          != null && <Stat label="Features"      value={m.n_features} />}
              {m.wf_decile_monotonicity != null && <Stat label="Decile Mono" value={fmt2(m.wf_decile_monotonicity)} />}
            </div>
          </Card>
        )}
      </div>

      {Object.keys(m).length > 0 && (
        <Card title="All Metrics">
          <pre className="json-pre">{JSON.stringify(m, null, 2)}</pre>
        </Card>
      )}
    </div>
  );
}

// ── ROOT APP ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage]               = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const core = usePageData(URLS.core,        { active: true,                                           refreshSec: 60 });
  const pred = usePageData(URLS.predictions, { active: page === "predictions" || page === "signal",    refreshSec: 120 });
  const sim  = usePageData(URLS.simulation,  { active: page === "simulation"  || page === "trades",    refreshSec: 120 });
  const mkt  = usePageData(URLS.market,      { active: page === "market",                              refreshSec: 120 });

  const anyLoading    = core.loading || pred.loading || sim.loading || mkt.loading;
  const overallStatus = core.data?.monitoring?.overall_status;

  const navGroups = [
    { label: "Main",   ids: ["overview", "predictions", "signal"] },
    { label: "Quant",  ids: ["simulation", "trades", "market"] },
    { label: "System", ids: ["pipeline", "model"] },
  ];

  function handleRefresh() {
    core.reload();
    if (page === "predictions" || page === "signal")  pred.reload();
    if (page === "simulation"  || page === "trades")  sim.reload();
    if (page === "market")                            mkt.reload();
  }

  return (
    <div className="shell">
      {/* ── Sidebar ── */}
      <nav className={`sidebar${sidebarOpen ? "" : " collapsed"}`}>
        <div className="sidebar-brand">
          <span className="brand-icon">◈</span>
          {sidebarOpen && <span className="brand-text">TalPred AI</span>}
        </div>

        {navGroups.map((g) => (
          <div key={g.label} className="nav-group">
            {sidebarOpen && <div className="nav-group-label">{g.label}</div>}
            {g.ids.map((id) => {
              const item = NAV.find((n) => n.id === id);
              if (!item) return null;
              return (
                <button key={id}
                  className={`nav-item${page === id ? " active" : ""}`}
                  onClick={() => setPage(id)}
                  title={item.label}>
                  <span className="nav-icon">{item.icon}</span>
                  {sidebarOpen && <span className="nav-label">{item.label}</span>}
                </button>
              );
            })}
          </div>
        ))}

        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)} title="Toggle sidebar">
          {sidebarOpen ? "‹" : "›"}
        </button>

        {sidebarOpen && (
          <div className="sidebar-footer">
            <StatusDot status={overallStatus ?? "unknown"} pulse={anyLoading} />
            <span className="sidebar-status">{overallStatus ?? "loading…"}</span>
          </div>
        )}
      </nav>

      {/* ── Main ── */}
      <div className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <span className="topbar-page">{NAV.find((n) => n.id === page)?.label ?? page}</span>
            {anyLoading && <Spinner />}
          </div>
          <div className="topbar-right">
            {core.data?.model?.model_version && (
              <span className="model-badge">{core.data.model.model_version}</span>
            )}
            <StatusDot status={overallStatus ?? "unknown"} />
            <span className="topbar-time muted">{core.lastUpdated ? ts(core.lastUpdated) : "—"}</span>
            <button className="reload-btn" onClick={handleRefresh}>↻ Refresh</button>
          </div>
        </header>

        <div className="page-area">
          {core.error && <ErrorBanner msg={`Core: ${core.error}`} />}
          {pred.error && (page === "predictions" || page === "signal") && <ErrorBanner msg={`Predictions: ${pred.error}`} />}
          {sim.error  && (page === "simulation"  || page === "trades") && <ErrorBanner msg={`Simulation: ${sim.error}`} />}
          {mkt.error  && page === "market" && <ErrorBanner msg={`Market: ${mkt.error}`} />}

          {page === "overview"    && <OverviewPage    coreData={core.data} simData={sim.data} predData={pred.data} mktData={mkt.data} />}
          {page === "predictions" && <PredictionsPage data={pred.data} />}
          {page === "signal"      && <SignalPage      data={pred.data} />}
          {page === "simulation"  && <SimulationPage  data={sim.data} />}
          {page === "trades"      && <TradesPage      data={sim.data} />}
          {page === "market"      && <MarketPage      data={mkt.data} />}
          {page === "pipeline"    && <PipelinePage    data={core.data} />}
          {page === "model"       && <ModelPage       data={core.data} />}
        </div>
      </div>
    </div>
  );
}
