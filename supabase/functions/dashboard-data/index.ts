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
    const d7 = isoDateDaysAgo(7);
    const ts7 = isoTimestampDaysAgo(7);

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
    ] = await Promise.all([
      countRows(sb, "symbols", "symbol", { eq: { is_active: true } }),
      selectRows(sb, "market_bars_daily", "date,symbol", { gte: { date: d30 } }),
      countRows(sb, "quarantine", "id", { gte: { created_at: ts7 } }),
      selectOne(sb, "market_bars_daily", "date", { orderDesc: "date" }),
      selectOne(sb, "features_snapshot", "target_session_date", { orderDesc: "target_session_date" }),
      selectOne(sb, "predictions", "as_of_time,target_date", { orderDesc: "as_of_time" }),
      selectRows(sb, "features_snapshot", "target_session_date", { gte: { target_session_date: d7 } }),
      selectRows(sb, "predictions", "target_date", { gte: { target_date: d7 } }),
      selectOne(sb, "model_registry", "model_version,algorithm,status,training_window_start,training_window_end,metrics", {
        eq: { status: "production" },
        orderDesc: "created_at",
      }),
      selectRows(
        sb,
        "predictions",
        "prediction_id,symbol,target_date,direction,probability_up,confidence,model_version",
        { gte: { confidence: 0.5 }, orderDesc: "as_of_time", limit: 100 },
      ),
    ]);

    const uniqueDays30 = new Set((bars30Rows ?? []).map((r) => r.date)).size;
    const totalBars30 = (bars30Rows ?? []).length;
    const expectedBars = uniqueDays30 * (activeSymbols ?? 0);
    const missingPct =
      expectedBars > 0 ? ((expectedBars - totalBars30) / expectedBars) * 100 : 0;

    const barDays7 = new Set((bars30Rows ?? []).filter((r) => r.date >= d7).map((r) => r.date)).size;
    const featDays7 = new Set((feat7Rows ?? []).map((r) => r.target_session_date)).size;
    const predDays7 = new Set((pred7Rows ?? []).map((r) => r.target_date)).size;

    const alerts: Array<{ level: string; check: string; detail: string }> = [];
    if (missingPct > 1.0) {
      alerts.push({
        level: "critical",
        check: "missing_bars",
        detail: `${missingPct.toFixed(2)}% bars missing in last 30 days (threshold: 1.0%)`,
      });
    }
    if ((quarantine7 ?? 0) > 3) {
      alerts.push({
        level: "warning",
        check: "ingestion_failures",
        detail: `${quarantine7} quarantine records in 7 days (threshold: 3)`,
      });
    }
    if (barDays7 > 0 && featDays7 === 0) {
      alerts.push({
        level: "critical",
        check: "feature_pipeline_stalled",
        detail: `Market data has ${barDays7} days but features have ${featDays7} days in last 7d`,
      });
    }
    if (featDays7 > 0 && predDays7 === 0) {
      alerts.push({
        level: "critical",
        check: "prediction_pipeline_stalled",
        detail: `Features have ${featDays7} days but predictions have ${predDays7} days in last 7d`,
      });
    }

    const monitoring = {
      timestamp: now.toISOString(),
      data_quality: {
        total_bars_30d: totalBars30,
        active_symbols: activeSymbols ?? 0,
        missing_pct: round(missingPct, 2),
        quarantine_7d: quarantine7 ?? 0,
        alerts: alerts.filter((a) => a.check === "missing_bars" || a.check === "ingestion_failures"),
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
        alerts: alerts.filter(
          (a) => a.check === "feature_pipeline_stalled" || a.check === "prediction_pipeline_stalled",
        ),
      },
      total_alerts: alerts.length,
      alert_details: alerts,
      overall_status: alerts.length === 0 ? "healthy" : "degraded",
    };

    const workflows = await fetchWorkflows();

    return json(200, {
      health: {
        status: "healthy",
        version: "edge-v1",
        cache_available: null,
        disclaimer: "Predictions are informational only and do not constitute financial advice.",
      },
      model: modelRow ?? null,
      monitoring,
      predictions: predRows ?? [],
      workflows,
    });
  } catch (err) {
    return json(500, { error: String(err) });
  }
});

async function fetchWorkflows(): Promise<{ available: boolean; reason?: string; runs: WorkflowRun[] }> {
  const token = Deno.env.get("GITHUB_TOKEN_DASHBOARD") || "";
  const owner = Deno.env.get("GITHUB_REPO_OWNER") || "";
  const repo = Deno.env.get("GITHUB_REPO_NAME") || "";
  if (!token || !owner || !repo) {
    return {
      available: false,
      reason: "Missing one of GITHUB_TOKEN_DASHBOARD, GITHUB_REPO_OWNER, GITHUB_REPO_NAME",
      runs: [],
    };
  }

  const files = [
    ".github/workflows/daily-pipeline.yml",
    ".github/workflows/monthly-archive.yml",
    ".github/workflows/news-sentiment.yml",
  ];
  const runs: WorkflowRun[] = [];
  for (const wf of files) {
    const url =
      `https://api.github.com/repos/${owner}/${repo}/actions/workflows/` +
      `${encodeURIComponent(wf)}/runs?per_page=1`;
    const resp = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
    if (!resp.ok) {
      const text = await resp.text();
      runs.push({
        workflow: wf,
        status: "error",
        conclusion: "error",
        message: `GitHub API error ${resp.status}: ${text.slice(0, 200)}`,
      });
      continue;
    }
    const data = await resp.json();
    const latest = data?.workflow_runs?.[0];
    if (!latest) {
      runs.push({ workflow: wf, status: "none", conclusion: "none" });
      continue;
    }
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

async function countRows(
  sb: ReturnType<typeof createClient>,
  table: string,
  column: string,
  opts: {
    eq?: Record<string, string | number | boolean>;
    gte?: Record<string, string | number>;
  } = {},
): Promise<number> {
  let q = sb.from(table).select(column, { count: "exact", head: true });
  if (opts.eq) {
    for (const [k, v] of Object.entries(opts.eq)) q = q.eq(k, v);
  }
  if (opts.gte) {
    for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  }
  const { count, error } = await q;
  if (error) throw error;
  return count ?? 0;
}

async function selectRows(
  sb: ReturnType<typeof createClient>,
  table: string,
  cols: string,
  opts: {
    eq?: Record<string, string | number | boolean>;
    gte?: Record<string, string | number>;
    orderDesc?: string;
    limit?: number;
  } = {},
): Promise<any[]> {
  let q = sb.from(table).select(cols);
  if (opts.eq) {
    for (const [k, v] of Object.entries(opts.eq)) q = q.eq(k, v);
  }
  if (opts.gte) {
    for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  }
  if (opts.orderDesc) {
    q = q.order(opts.orderDesc, { ascending: false });
  }
  if (opts.limit) {
    q = q.limit(opts.limit);
  }
  const { data, error } = await q;
  if (error) throw error;
  return data ?? [];
}

async function selectOne(
  sb: ReturnType<typeof createClient>,
  table: string,
  cols: string,
  opts: {
    eq?: Record<string, string | number | boolean>;
    orderDesc?: string;
  } = {},
): Promise<any | null> {
  const rows = await selectRows(sb, table, cols, {
    eq: opts.eq,
    orderDesc: opts.orderDesc,
    limit: 1,
  });
  return rows[0] ?? null;
}

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
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
