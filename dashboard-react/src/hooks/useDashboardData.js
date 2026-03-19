import { useCallback, useEffect, useState } from "react";

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
  return resp.json();
}

const EMPTY = {
  health: null,
  model: null,
  monitoring: null,
  predictions: [],
  workflows: null,
  simulation: { runs: [], latest_trades: [], latest_run_metrics: {} },
  signal_health: { ic_mean: null, ic_30d: null, ic_prior_30d: null, ic_trend: "unknown", n_ic_days: 0, decile_spread_bps: null, ic_series: [] },
  market_context: { vix: null, regime: null, sp500_momentum_200d: null, spy_drawdown_60d: null, spy_series: [], spy_dates: [] },
  alpha_quality: { hit_rates: {}, n_outcomes_30d: 0, n_outcomes_120d: 0 },
};

export function useDashboardData(refreshSec = 30) {
  const [data, setData] = useState(EMPTY);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const reload = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const edgeUrl = import.meta.env.VITE_DASHBOARD_EDGE_URL;
      if (!edgeUrl) throw new Error("Missing VITE_DASHBOARD_EDGE_URL");
      const payload = await fetchJson(edgeUrl);
      setData({
        health:         payload.health         ?? EMPTY.health,
        model:          payload.model          ?? EMPTY.model,
        monitoring:     payload.monitoring     ?? EMPTY.monitoring,
        predictions:    payload.predictions    ?? [],
        workflows:      payload.workflows      ?? { available: false, runs: [] },
        simulation:     payload.simulation     ?? EMPTY.simulation,
        signal_health:  payload.signal_health  ?? EMPTY.signal_health,
        market_context: payload.market_context ?? EMPTY.market_context,
        alpha_quality:  payload.alpha_quality  ?? EMPTY.alpha_quality,
      });
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { reload(); }, [reload]);

  useEffect(() => {
    const ms = Math.max(5, Number(refreshSec)) * 1000;
    const t = setInterval(reload, ms);
    return () => clearInterval(t);
  }, [refreshSec, reload]);

  return { data, loading, error, reload, lastUpdated };
}
