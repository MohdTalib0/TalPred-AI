import { createClient } from "npm:@supabase/supabase-js@2.49.1";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  try {
    const sb = makeSB();
    const d7  = daysAgo(7);
    const d30 = daysAgo(30);
    const ts7 = tsAgo(7);

    const [
      activeSymbols, bars30Rows, quarantine7,
      latestBarRow, latestFeatureRow, latestPredictionRow,
      feat7Rows, pred7Rows, modelRow,
    ] = await Promise.all([
      countRows(sb, "symbols",         "symbol",              { eq:  { is_active: true } }),
      selectRows(sb, "market_bars_daily", "date,symbol",      { gte: { date: d30 } }),
      countRows(sb, "quarantine",      "id",                  { gte: { created_at: ts7 } }),
      selectOne(sb,  "market_bars_daily", "date",             { orderDesc: "date" }),
      selectOne(sb,  "features_snapshot", "target_session_date", { orderDesc: "target_session_date" }),
      selectOne(sb,  "predictions",    "as_of_time,target_date", { orderDesc: "as_of_time" }),
      selectRows(sb, "features_snapshot","target_session_date",  { gte: { target_session_date: d7 } }),
      selectRows(sb, "predictions",    "target_date",         { gte: { target_date: d7 } }),
      selectOne(sb,  "model_registry",
        "model_version,algorithm,status,training_window_start,training_window_end,metrics,created_at,promoted_at",
        { eq: { status: "production" }, orderDesc: "created_at" }),
    ]);

    const uniqueDays30  = new Set((bars30Rows ?? []).map((r) => r.date)).size;
    const totalBars30   = (bars30Rows ?? []).length;
    const expectedBars  = uniqueDays30 * (activeSymbols ?? 0);
    const missingPct    = expectedBars > 0 ? ((expectedBars - totalBars30) / expectedBars) * 100 : 0;
    const barDays7      = new Set((bars30Rows ?? []).filter((r) => r.date >= d7).map((r) => r.date)).size;
    const featDays7     = new Set((feat7Rows ?? []).map((r) => r.target_session_date)).size;
    const predDays7     = new Set((pred7Rows ?? []).map((r) => r.target_date)).size;

    const alerts: Array<{ level: string; check: string; detail: string }> = [];
    if (missingPct > 1.0)
      alerts.push({ level: "critical", check: "missing_bars",           detail: `${missingPct.toFixed(2)}% bars missing (threshold 1%)` });
    if ((quarantine7 ?? 0) > 3)
      alerts.push({ level: "warning",  check: "ingestion_failures",     detail: `${quarantine7} quarantine records in 7d (threshold 3)` });
    if (barDays7 > 0 && featDays7 === 0)
      alerts.push({ level: "critical", check: "feature_pipeline_stalled", detail: `${barDays7} bar-days but 0 feature-days in last 7d` });
    if (featDays7 > 0 && predDays7 === 0)
      alerts.push({ level: "critical", check: "prediction_pipeline_stalled", detail: `${featDays7} feature-days but 0 prediction-days in last 7d` });

    const monitoring = {
      timestamp: new Date().toISOString(),
      overall_status: alerts.length === 0 ? "healthy" : "degraded",
      total_alerts: alerts.length,
      alert_details: alerts,
      data_quality: {
        total_bars_30d: totalBars30,
        active_symbols: activeSymbols ?? 0,
        missing_pct: round(missingPct, 2),
        quarantine_7d: quarantine7 ?? 0,
      },
      data_freshness: {
        latest_market_bar:   latestBarRow?.date ?? null,
        latest_feature:      latestFeatureRow?.target_session_date ?? null,
        latest_prediction:   latestPredictionRow?.as_of_time ?? null,
      },
      pipeline_health: {
        market_bar_days_7d: barDays7,
        feature_days_7d:    featDays7,
        prediction_days_7d: predDays7,
      },
    };

    const workflows = await fetchWorkflows();

    return json(200, {
      health: {
        status: "healthy",
        version: "edge-v3",
        disclaimer: "Predictions are informational only and do not constitute financial advice.",
      },
      model: modelRow ?? null,
      monitoring,
      workflows,
    });
  } catch (err) {
    return json(500, { error: fmtErr(err) });
  }
});

// ── GitHub ────────────────────────────────────────────────────────────────────
async function fetchWorkflows() {
  const token = Deno.env.get("GITHUB_TOKEN_DASHBOARD") || "";
  const owner = Deno.env.get("GITHUB_REPO_OWNER") || "";
  const repo  = Deno.env.get("GITHUB_REPO_NAME")  || "";
  if (!token || !owner || !repo)
    return { available: false, reason: "Missing GITHUB_TOKEN_DASHBOARD / GITHUB_REPO_OWNER / GITHUB_REPO_NAME", runs: [] };

  const files = [
    ".github/workflows/daily-pipeline.yml",
    ".github/workflows/monthly-archive.yml",
    ".github/workflows/news-sentiment.yml",
  ];
  const runs = [];
  for (const wf of files) {
    const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(wf)}/runs?per_page=1`;
    const resp = await fetch(url, {
      headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" },
    });
    if (!resp.ok) {
      runs.push({ workflow: wf, status: "error", conclusion: "error", message: `GitHub ${resp.status}` });
      continue;
    }
    const d = await resp.json();
    const r = d?.workflow_runs?.[0];
    if (!r) { runs.push({ workflow: wf, status: "none", conclusion: "none" }); continue; }
    runs.push({ workflow: wf, status: r.status, conclusion: r.conclusion || "in_progress",
      run_number: r.run_number, event: r.event, created_at: r.created_at, updated_at: r.updated_at, html_url: r.html_url });
  }
  return { available: true, runs };
}

// ── Supabase helpers ──────────────────────────────────────────────────────────
function makeSB() {
  const url = Deno.env.get("SUPABASE_URL");
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !key) throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, key, { auth: { persistSession: false } });
}
type SB = ReturnType<typeof makeSB>;
type Opts = { eq?: Record<string, unknown>; gte?: Record<string, unknown>; orderDesc?: string; limit?: number };

async function countRows(sb: SB, table: string, col: string, opts: Opts = {}): Promise<number> {
  let q = sb.from(table).select(col, { count: "exact", head: true });
  if (opts.eq)  for (const [k, v] of Object.entries(opts.eq))  q = q.eq(k, v);
  if (opts.gte) for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  const { count, error } = await q;
  if (error) { if (error.code === "57014") return 0; throw new Error(`[${table}] ${error.message}`); }
  return count ?? 0;
}
async function selectRows(sb: SB, table: string, cols: string, opts: Opts = {}): Promise<any[]> {
  let q = sb.from(table).select(cols);
  if (opts.eq)       for (const [k, v] of Object.entries(opts.eq))  q = q.eq(k, v);
  if (opts.gte)      for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  if (opts.orderDesc) q = q.order(opts.orderDesc, { ascending: false });
  if (opts.limit)     q = q.limit(opts.limit);
  const { data, error } = await q;
  if (error) { if (error.code === "57014") return []; throw new Error(`[${table}] ${error.message}`); }
  return data ?? [];
}
async function selectOne(sb: SB, table: string, cols: string, opts: Opts = {}): Promise<any | null> {
  return (await selectRows(sb, table, cols, { ...opts, limit: 1 }))[0] ?? null;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status, headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
function daysAgo(n: number): string { const d = new Date(); d.setUTCDate(d.getUTCDate() - n); return d.toISOString().slice(0, 10); }
function tsAgo(n: number): string   { const d = new Date(); d.setUTCDate(d.getUTCDate() - n); return d.toISOString(); }
function round(v: number, n: number): number { const p = 10 ** n; return Math.round(v * p) / p; }
function fmtErr(e: unknown): string { if (e instanceof Error) return e.message; try { return JSON.stringify(e); } catch { return String(e); } }
