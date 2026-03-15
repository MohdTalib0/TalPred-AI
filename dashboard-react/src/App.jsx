import { useMemo, useState } from "react";
import { useDashboardData } from "./hooks/useDashboardData";

const REFRESH_OPTIONS = [15, 30, 60];

export default function App() {
  const [refreshSec, setRefreshSec] = useState(30);
  const { data, loading, error, reload, lastUpdated } = useDashboardData(refreshSec);

  const predictionStats = useMemo(() => {
    const rows = data.predictions || [];
    if (rows.length === 0) {
      return { total: 0, avgConf: null, up: 0, down: 0 };
    }
    const up = rows.filter((r) => r.direction === "up").length;
    const down = rows.filter((r) => r.direction === "down").length;
    const avgConf = rows.reduce((a, r) => a + Number(r.confidence || 0), 0) / rows.length;
    return { total: rows.length, avgConf, up, down };
  }, [data.predictions]);

  const allAlerts = data.monitoring?.alert_details || [];
  const criticalCount = allAlerts.filter((a) => a.level === "critical").length;
  const warningCount = allAlerts.filter((a) => a.level === "warning").length;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>TalPred Operations Dashboard</h1>
          <p>
            Live model activity, pipeline health, archive workflow status, and prediction feed.
          </p>
        </div>
        <div className="topbar-actions">
          <label>
            Auto-refresh
            <select
              value={refreshSec}
              onChange={(e) => setRefreshSec(Number(e.target.value))}
            >
              {REFRESH_OPTIONS.map((v) => (
                <option key={v} value={v}>
                  {v}s
                </option>
              ))}
            </select>
          </label>
          <button onClick={reload} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh now"}
          </button>
          <span className="muted">
            Last updated: {lastUpdated ? new Date(lastUpdated).toLocaleString() : "-"}
          </span>
        </div>
      </header>

      {error && <div className="banner error">Data refresh failed: {error}</div>}

      <section className="grid cards-4">
        <StatCard
          title="API Health"
          value={data.health?.status || "unknown"}
          tone={data.health?.status === "healthy" ? "ok" : "warn"}
          meta={`Cache: ${data.health?.cache_available ? "available" : "unavailable"}`}
        />
        <StatCard
          title="Production Model"
          value={data.model?.model_version || "unavailable"}
          tone={data.model ? "ok" : "warn"}
          meta={`${data.model?.algorithm || "unknown"} | ${data.model?.status || "-"}`}
        />
        <StatCard
          title="Monitoring"
          value={data.monitoring?.overall_status || "unknown"}
          tone={data.monitoring?.overall_status === "healthy" ? "ok" : "warn"}
          meta={`critical=${criticalCount}, warning=${warningCount}`}
        />
        <StatCard
          title="Prediction Feed"
          value={predictionStats.total}
          tone="ok"
          meta={
            predictionStats.total > 0
              ? `avg conf=${predictionStats.avgConf?.toFixed(3)} | up=${predictionStats.up} down=${predictionStats.down}`
              : "No recent rows"
          }
        />
      </section>

      <section className="grid cards-3">
        <MetricPanel title="Data Quality" data={data.monitoring?.data_quality} />
        <MetricPanel title="Freshness" data={data.monitoring?.data_freshness} />
        <MetricPanel title="Pipeline Health" data={data.monitoring?.pipeline_health} />
      </section>

      <section className="grid cards-2">
        <article className="card">
          <h2>Alert Feed</h2>
          {allAlerts.length === 0 ? (
            <p className="ok">No active alerts.</p>
          ) : (
            <ul className="list">
              {allAlerts.slice(0, 20).map((a, idx) => (
                <li key={`${a.check}-${idx}`} className={`list-item ${a.level === "critical" ? "bad" : "warn"}`}>
                  <div className="row">
                    <strong>{a.level.toUpperCase()}</strong>
                    <span>{a.check}</span>
                  </div>
                  <small>{a.detail}</small>
                </li>
              ))}
            </ul>
          )}
        </article>

        <article className="card">
          <h2>Workflow Status</h2>
          {!data.workflows?.available ? (
            <p className="warn">{data.workflows?.reason || "Workflow data unavailable."}</p>
          ) : (
            <ul className="list">
              {(data.workflows?.runs || []).map((r) => {
                const tone =
                  r.conclusion === "success"
                    ? "ok"
                    : r.conclusion === "failure" || r.conclusion === "error"
                      ? "bad"
                      : "warn";
                return (
                  <li key={r.workflow} className={`list-item ${tone}`}>
                    <div className="row">
                      <strong>{r.workflow.replace(".github/workflows/", "")}</strong>
                      <span>{r.conclusion || r.status}</span>
                    </div>
                    <small>
                      #{r.run_number ?? "-"} | {r.event || "-"} |{" "}
                      {r.updated_at ? new Date(r.updated_at).toLocaleString() : "-"}{" "}
                      {r.html_url ? (
                        <>
                          |{" "}
                          <a href={r.html_url} target="_blank" rel="noreferrer">
                            open
                          </a>
                        </>
                      ) : null}
                    </small>
                  </li>
                );
              })}
            </ul>
          )}
        </article>
      </section>

      <section className="card">
        <h2>Latest Predictions</h2>
        <PredictionTable rows={data.predictions || []} />
      </section>
    </div>
  );
}

function StatCard({ title, value, tone, meta }) {
  return (
    <article className="card">
      <h2>{title}</h2>
      <div className={`value ${tone}`}>{String(value)}</div>
      <small>{meta}</small>
    </article>
  );
}

function MetricPanel({ title, data }) {
  return (
    <article className="card">
      <h2>{title}</h2>
      {!data ? (
        <p className="muted">No data</p>
      ) : (
        <div className="kv-grid">
          {Object.entries(data)
            .filter(([k]) => k !== "alerts")
            .map(([k, v]) => (
              <div key={k} className="kv-item">
                <span className="muted">{k}</span>
                <strong>{fmt(v)}</strong>
              </div>
            ))}
        </div>
      )}
    </article>
  );
}

function PredictionTable({ rows }) {
  if (rows.length === 0) {
    return <p className="muted">No prediction rows returned.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Target Date</th>
            <th>Direction</th>
            <th>Prob Up</th>
            <th>Confidence</th>
            <th>Model</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.prediction_id}>
              <td>{r.symbol}</td>
              <td>{r.target_date}</td>
              <td className={r.direction === "up" ? "ok" : "warn"}>{r.direction}</td>
              <td>{Number(r.probability_up).toFixed(3)}</td>
              <td>{Number(r.confidence).toFixed(3)}</td>
              <td>{r.model_version || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : String(v);
  return String(v);
}
