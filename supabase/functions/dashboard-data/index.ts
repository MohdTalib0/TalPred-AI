import { createClient } from "npm:@supabase/supabase-js@2.49.1";

type WorkflowRun = {
  workflow: string;
  status: string;
  conclusion: string;
  run_number?: number;
  event?: string;
  created_at?: string;
  updated_at?: string;
  html_url?: string;
  message?: string;
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceRole = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !serviceRole) {
      return json(500, { error: "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY" });
    }

    const sb = createClient(supabaseUrl, serviceRole, {
      auth: { persistSession: false },
    });

    const now = new Date();
    const d30 = isoDateDaysAgo(30);
    const d60 = isoDateDaysAgo(60);
    const d120 = isoDateDaysAgo(120);
    const d7 = isoDateDaysAgo(7);
    const ts7 = isoTimestampDaysAgo(7);

    // ── Parallel batch 1: all independent queries ────────────────────
    const [
      activeSymbols,
      bars30Rows,
      quarantine7,
      latestBarRow,
      latestFeatureRow,
      latestPredictionRow,
      feat7Rows,
      pred7Rows,
      modelRow,
      predRows,
      simRunRows,
      spyRows,
      vixRow,
      predIcRows,
      tradeRows,
    ] = await Promise.all([
      countRows(sb, "symbols", "symbol", { eq: { is_active: true } }),
      selectRows(sb, "market_bars_daily", "date,symbol", { gte: { date: d30 } }),
      countRows(sb, "quarantine", "id", { gte: { created_at: ts7 } }),
      selectOne(sb, "market_bars_daily", "date", { orderDesc: "date" }),
      selectOne(sb, "features_snapshot", "target_session_date", { orderDesc: "target_session_date" }),
      selectOne(sb, "predictions", "as_of_time,target_date", { orderDesc: "as_of_time" }),
      selectRows(sb, "features_snapshot", "target_session_date", { gte: { target_session_date: d7 } }),
      selectRows(sb, "predictions", "target_date", { gte: { target_date: d7 } }),
      selectOne(sb, "model_registry",
        "model_version,algorithm,status,training_window_start,training_window_end,metrics,created_at",
        { eq: { status: "production" }, orderDesc: "created_at" }
      ),
      // Enriched predictions for display (with outcome fields)
      selectRows(sb, "predictions",
        "prediction_id,symbol,target_date,direction,probability_up,confidence,model_version,realized_direction,realized_return",
        { gte: { confidence: 0.5 }, orderDesc: "as_of_time", limit: 200 }
      ),
      // Recent simulation runs
      selectRows(sb, "simulation_runs",
        "run_id,created_at,start_date,end_date,strategy_name,status,result_metrics,starting_capital,model_version",
        { eq: { status: "completed" }, orderDesc: "created_at", limit: 10 }
      ),
      // SPY bars for drawdown + sparkline
      selectRows(sb, "market_bars_daily",
        "date,close",
        { eq: { symbol: "SPY" }, gte: { date: d60 }, orderDesc: "date", limit: 70 }
      ),
      // Latest VIX + regime from features_snapshot
      selectOne(sb, "features_snapshot",
        "vix_level,sp500_momentum_200d,regime_label,target_session_date",
        { orderDesc: "target_session_date" }
      ),
      // Historical predictions with outcomes for signal health computation
      selectRows(sb, "predictions",
        "target_date,probability_up,confidence,direction,realized_direction,realized_return",
        { gte: { target_date: d120 }, orderDesc: "target_date", limit: 5000 }
      ),
      // Recent paper trades
      selectRows(sb, "paper_trades",
        "run_id,date,symbol,weight,entry_price,exit_price,daily_pnl,transaction_cost,slippage_cost",
        { orderDesc: "date", limit: 60 }
      ),
    ]);

    // ── Compute monitoring aggregates ────────────────────────────────
    const uniqueDays30 = new Set((bars30Rows ?? []).map((r) => r.date)).size;
    const totalBars30 = (bars30Rows ?? []).length;
    const expectedBars = uniqueDays30 * (activeSymbols ?? 0);
    const missingPct = expectedBars > 0 ? ((expectedBars - totalBars30) / expectedBars) * 100 : 0;
    const barDays7 = new Set((bars30Rows ?? []).filter((r) => r.date >= d7).map((r) => r.date)).size;
    const featDays7 = new Set((feat7Rows ?? []).map((r) => r.target_session_date)).size;
    const predDays7 = new Set((pred7Rows ?? []).map((r) => r.target_date)).size;

    const alerts: Array<{ level: string; check: string; detail: string }> = [];
    if (missingPct > 1.0) alerts.push({ level: "critical", check: "missing_bars", detail: `${missingPct.toFixed(2)}% bars missing in last 30 days (threshold: 1.0%)` });
    if ((quarantine7 ?? 0) > 3) alerts.push({ level: "warning", check: "ingestion_failures", detail: `${quarantine7} quarantine records in 7 days (threshold: 3)` });
    if (barDays7 > 0 && featDays7 === 0) alerts.push({ level: "critical", check: "feature_pipeline_stalled", detail: `Market data has ${barDays7} days but features have 0 days in last 7d` });
    if (featDays7 > 0 && predDays7 === 0) alerts.push({ level: "critical", check: "prediction_pipeline_stalled", detail: `Features have ${featDays7} days but predictions have 0 days in last 7d` });

    const monitoring = {
      timestamp: now.toISOString(),
      data_quality: {
        total_bars_30d: totalBars30,
        active_symbols: activeSymbols ?? 0,
        missing_pct: round(missingPct, 2),
        quarantine_7d: quarantine7 ?? 0,
        alerts: alerts.filter((a) => ["missing_bars", "ingestion_failures"].includes(a.check)),
      },
      data_freshness: {
        latest_market_bar: latestBarRow?.date ?? null,
        latest_feature: latestFeatureRow?.target_session_date ?? null,
        latest_prediction: latestPredictionRow?.as_of_time ?? null,
        alerts: [],
      },
      pipeline_health: {
        market_bar_days_7d: barDays7,
        feature_days_7d: featDays7,
        prediction_days_7d: predDays7,
        alerts: alerts.filter((a) => ["feature_pipeline_stalled", "prediction_pipeline_stalled"].includes(a.check)),
      },
      total_alerts: alerts.length,
      alert_details: alerts,
      overall_status: alerts.length === 0 ? "healthy" : "degraded",
    };

    // ── Signal health from historical predictions ────────────────────
    const withOutcomes = (predIcRows ?? []).filter(
      (p) => p.realized_return != null && p.probability_up != null
    );
    const signal_health = computeSignalHealth(withOutcomes, d30);

    // ── Hit rates by confidence tier ─────────────────────────────────
    const hit_rates = computeHitRates(withOutcomes);

    // ── Market context from VIX row + SPY bars ───────────────────────
    const spySorted = [...(spyRows ?? [])].sort((a, b) => a.date.localeCompare(b.date));
    const spyCloses = spySorted.map((r) => r.close);
    const spyDrawdown = computeSPYDrawdown(spyCloses);

    const market_context = {
      vix: vixRow?.vix_level ? round(vixRow.vix_level, 1) : null,
      regime: vixRow?.regime_label ?? null,
      sp500_momentum_200d: vixRow?.sp500_momentum_200d ? round(vixRow.sp500_momentum_200d * 100, 2) : null,
      spy_drawdown_60d: spyDrawdown.drawdown,
      spy_series: spyCloses.slice(-60).map((c) => round(c, 2)),
      spy_dates: spySorted.slice(-60).map((r) => r.date),
    };

    // ── Simulation runs + performance ────────────────────────────────
    const runs = (simRunRows ?? []).map((r) => ({
      run_id: r.run_id,
      created_at: r.created_at,
      start_date: r.start_date,
      end_date: r.end_date,
      strategy_name: r.strategy_name ?? "legacy",
      model_version: r.model_version,
      metrics: r.result_metrics ?? {},
    }));

    const latestRunMetrics = runs[0]?.metrics ?? {};

    const workflows = await fetchWorkflows();

    return json(200, {
      health: {
        status: "healthy",
        version: "edge-v2",
        cache_available: null,
        disclaimer: "Predictions are informational only and do not constitute financial advice.",
      },
      model: modelRow ?? null,
      monitoring,
      predictions: predRows ?? [],
      workflows,
      simulation: {
        runs,
        latest_trades: tradeRows ?? [],
        latest_run_metrics: latestRunMetrics,
      },
      signal_health,
      market_context,
      alpha_quality: {
        hit_rates,
        n_outcomes_30d: withOutcomes.filter((p) => p.target_date >= d30).length,
        n_outcomes_120d: withOutcomes.length,
      },
    });
  } catch (err) {
    return json(500, { error: formatError(err) });
  }
});

// ── Signal health computation ─────────────────────────────────────────

type PredRow = {
  target_date: string;
  probability_up: number;
  confidence?: number;
  direction?: string;
  realized_direction?: string;
  realized_return: number;
};

function computeSignalHealth(preds: PredRow[], d30: string) {
  // Group by date
  const byDate: Record<string, { prob: number; ret: number }[]> = {};
  for (const p of preds) {
    if (!byDate[p.target_date]) byDate[p.target_date] = [];
    byDate[p.target_date].push({ prob: p.probability_up, ret: p.realized_return });
  }

  const dailyIC: { date: string; ic: number; spreadBps: number }[] = [];
  for (const [date, items] of Object.entries(byDate)) {
    if (items.length < 10) continue;
    const ic = spearmanIC(items.map((i) => i.prob), items.map((i) => i.ret));
    const sorted = [...items].sort((a, b) => b.prob - a.prob);
    const n5 = Math.max(1, Math.floor(sorted.length / 5));
    const topRet = sorted.slice(0, n5).reduce((s, i) => s + i.ret, 0) / n5;
    const botRet = sorted.slice(-n5).reduce((s, i) => s + i.ret, 0) / n5;
    dailyIC.push({ date, ic, spreadBps: (topRet - botRet) * 10000 });
  }
  dailyIC.sort((a, b) => a.date.localeCompare(b.date));

  if (dailyIC.length === 0) {
    return { ic_mean: null, ic_30d: null, ic_prior_30d: null, ic_trend: "unknown", n_ic_days: 0, decile_spread_bps: null, ic_series: [] };
  }

  const recent = dailyIC.filter((d) => d.date >= d30);
  const prior = dailyIC.filter((d) => d.date < d30);
  const ic_mean = mean(dailyIC.map((d) => d.ic));
  const ic_30d = recent.length > 0 ? mean(recent.map((d) => d.ic)) : null;
  const ic_prior_30d = prior.length > 0 ? mean(prior.map((d) => d.ic)) : null;
  const decile_spread_bps = dailyIC.length > 0 ? mean(dailyIC.slice(-30).map((d) => d.spreadBps)) : null;

  return {
    ic_mean: round(ic_mean, 4),
    ic_30d: ic_30d !== null ? round(ic_30d, 4) : null,
    ic_prior_30d: ic_prior_30d !== null ? round(ic_prior_30d, 4) : null,
    ic_trend: ic_30d !== null && ic_prior_30d !== null
      ? (ic_30d > ic_prior_30d ? "improving" : "declining")
      : "unknown",
    n_ic_days: dailyIC.length,
    decile_spread_bps: decile_spread_bps !== null ? round(decile_spread_bps, 1) : null,
    ic_series: dailyIC.slice(-60).map((d) => ({ date: d.date, ic: round(d.ic, 4), spread: round(d.spreadBps, 1) })),
  };
}

function computeHitRates(preds: PredRow[]) {
  const tiers = [
    { label: "50–55%", lo: 0.50, hi: 0.55 },
    { label: "55–60%", lo: 0.55, hi: 0.60 },
    { label: "60–70%", lo: 0.60, hi: 0.70 },
    { label: "70%+",   lo: 0.70, hi: 1.01 },
  ];
  const result: Record<string, { n: number; hit_rate: number }> = {};
  for (const tier of tiers) {
    const filtered = preds.filter(
      (p) => p.realized_direction != null && (p.confidence ?? p.probability_up) >= tier.lo && (p.confidence ?? p.probability_up) < tier.hi
    );
    if (filtered.length < 5) continue;
    const correct = filtered.filter((p) => p.direction === p.realized_direction).length;
    result[tier.label] = { n: filtered.length, hit_rate: round(correct / filtered.length, 3) };
  }
  return result;
}

function computeSPYDrawdown(closes: number[]) {
  if (closes.length === 0) return { drawdown: null };
  let peak = closes[0];
  let dd = 0;
  for (const c of closes) {
    if (c > peak) peak = c;
    dd = (c - peak) / peak;
  }
  return { drawdown: round(dd, 4) };
}

// ── Math helpers ──────────────────────────────────────────────────────

function spearmanIC(probs: number[], rets: number[]): number {
  return pearson(rankArray(probs), rankArray(rets));
}

function rankArray(arr: number[]): number[] {
  const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const result = new Array(arr.length);
  for (let i = 0; i < indexed.length; i++) result[indexed[i].i] = i + 1;
  return result;
}

function pearson(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    dx += (x[i] - mx) ** 2;
    dy += (y[i] - my) ** 2;
  }
  return dx > 0 && dy > 0 ? num / Math.sqrt(dx * dy) : 0;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

// ── GitHub workflows ──────────────────────────────────────────────────

async function fetchWorkflows(): Promise<{ available: boolean; reason?: string; runs: WorkflowRun[] }> {
  const token = Deno.env.get("GITHUB_TOKEN_DASHBOARD") || "";
  const owner = Deno.env.get("GITHUB_REPO_OWNER") || "";
  const repo = Deno.env.get("GITHUB_REPO_NAME") || "";
  if (!token || !owner || !repo) {
    return { available: false, reason: "Missing GITHUB_TOKEN_DASHBOARD, GITHUB_REPO_OWNER, or GITHUB_REPO_NAME", runs: [] };
  }

  const files = [
    ".github/workflows/daily-pipeline.yml",
    ".github/workflows/monthly-archive.yml",
    ".github/workflows/news-sentiment.yml",
  ];
  const runs: WorkflowRun[] = [];
  for (const wf of files) {
    const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(wf)}/runs?per_page=1`;
    const resp = await fetch(url, {
      headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" },
    });
    if (!resp.ok) {
      const text = await resp.text();
      runs.push({ workflow: wf, status: "error", conclusion: "error", message: `GitHub API ${resp.status}: ${text.slice(0, 200)}` });
      continue;
    }
    const data = await resp.json();
    const latest = data?.workflow_runs?.[0];
    if (!latest) { runs.push({ workflow: wf, status: "none", conclusion: "none" }); continue; }
    runs.push({
      workflow: wf,
      status: latest.status,
      conclusion: latest.conclusion || "in_progress",
      run_number: latest.run_number,
      event: latest.event,
      created_at: latest.created_at,
      updated_at: latest.updated_at,
      html_url: latest.html_url,
    });
  }
  return { available: true, runs };
}

// ── Supabase helpers ──────────────────────────────────────────────────

async function countRows(
  sb: ReturnType<typeof createClient>,
  table: string,
  column: string,
  opts: { eq?: Record<string, string | number | boolean>; gte?: Record<string, string | number> } = {},
): Promise<number> {
  let q = sb.from(table).select(column, { count: "exact", head: true });
  if (opts.eq) for (const [k, v] of Object.entries(opts.eq)) q = q.eq(k, v);
  if (opts.gte) for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  const { count, error } = await q;
  if (error) {
    if (error.code === "57014") { console.warn(`[countRows:${table}] timeout`); return 0; }
    throw new Error(`[countRows:${table}] ${error.message}`);
  }
  return count ?? 0;
}

async function selectRows(
  sb: ReturnType<typeof createClient>,
  table: string,
  cols: string,
  opts: { eq?: Record<string, string | number | boolean>; gte?: Record<string, string | number>; orderDesc?: string; limit?: number } = {},
): Promise<any[]> {
  let q = sb.from(table).select(cols);
  if (opts.eq) for (const [k, v] of Object.entries(opts.eq)) q = q.eq(k, v);
  if (opts.gte) for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  if (opts.orderDesc) q = q.order(opts.orderDesc, { ascending: false });
  if (opts.limit) q = q.limit(opts.limit);
  const { data, error } = await q;
  if (error) {
    if (error.code === "57014") { console.warn(`[selectRows:${table}] timeout`); return []; }
    throw new Error(`[selectRows:${table}] ${error.message}`);
  }
  return data ?? [];
}

async function selectOne(
  sb: ReturnType<typeof createClient>,
  table: string,
  cols: string,
  opts: { eq?: Record<string, string | number | boolean>; orderDesc?: string } = {},
): Promise<any | null> {
  const rows = await selectRows(sb, table, cols, { eq: opts.eq, orderDesc: opts.orderDesc, limit: 1 });
  return rows[0] ?? null;
}

// ── Utilities ─────────────────────────────────────────────────────────

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function isoDateDaysAgo(days: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

function isoTimestampDaysAgo(days: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString();
}

function round(v: number, n: number): number {
  const p = 10 ** n;
  return Math.round(v * p) / p;
}

function formatError(err: unknown): string {
  if (err instanceof Error) return err.message;
  try { return JSON.stringify(err); } catch { return String(err); }
}
