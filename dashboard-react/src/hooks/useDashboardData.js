import { useCallback, useEffect, useState } from "react";

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`${url} -> HTTP ${resp.status}`);
  }
  return resp.json();
}

export function useDashboardData(refreshSec = 30) {
  const [data, setData] = useState({
    health: null,
    model: null,
    monitoring: null,
    predictions: [],
    workflows: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const reload = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const edgeUrl = import.meta.env.VITE_DASHBOARD_EDGE_URL;
      if (!edgeUrl) {
        throw new Error("Missing VITE_DASHBOARD_EDGE_URL");
      }
      const payload = await fetchJson(edgeUrl);
      setData({
        health: payload.health ?? null,
        model: payload.model ?? null,
        monitoring: payload.monitoring ?? null,
        predictions: payload.predictions ?? [],
        workflows: payload.workflows ?? { available: false, runs: [] },
      });
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  useEffect(() => {
    const ms = Math.max(5, Number(refreshSec)) * 1000;
    const t = setInterval(() => {
      reload();
    }, ms);
    return () => clearInterval(t);
  }, [refreshSec, reload]);

  return { data, loading, error, reload, lastUpdated };
}
