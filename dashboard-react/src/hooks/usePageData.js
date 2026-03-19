import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Lazy-loading per-page data hook.
 * Data is only fetched the first time `active` becomes true,
 * then cached until the component unmounts or `reload()` is called.
 */
export function usePageData(url, { active = true, refreshSec = 0 } = {}) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [loaded, setLoaded]   = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");
  const loadedRef = useRef(false);

  const reload = useCallback(async () => {
    if (!url) return;
    setLoading(true);
    setError("");
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
      const payload = await resp.json();
      setData(payload);
      setLoaded(true);
      loadedRef.current = true;
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [url]);

  // Fetch on first activation
  useEffect(() => {
    if (active && !loadedRef.current) reload();
  }, [active, reload]);

  // Auto-refresh
  useEffect(() => {
    if (!refreshSec || refreshSec < 5) return;
    const t = setInterval(reload, refreshSec * 1000);
    return () => clearInterval(t);
  }, [refreshSec, reload]);

  return { data, loading, error, loaded, lastUpdated, reload };
}
