import { createClient } from "npm:@supabase/supabase-js@2.49.1";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  try {
    const sb   = makeSB();
    const d30  = daysAgo(30);
    const d120 = daysAgo(120);

    const [predRows, predIcRows] = await Promise.all([
      // Latest 200 predictions for display table
      selectRows(sb, "predictions",
        "prediction_id,symbol,target_date,direction,probability_up,confidence,model_version,realized_direction,realized_return",
        { orderDesc: "as_of_time", limit: 200 }),
      // 120d of predictions with outcomes for signal health
      selectRows(sb, "predictions",
        "target_date,probability_up,confidence,direction,realized_direction,realized_return",
        { gte: { target_date: d120 }, orderDesc: "target_date", limit: 5000 }),
    ]);

    const withOutcomes = (predIcRows ?? []).filter(
      (p) => p.realized_return != null && p.probability_up != null
    );

    const signal_health = computeSignalHealth(withOutcomes, d30);
    const hit_rates     = computeHitRates(withOutcomes);

    return json(200, {
      predictions: predRows ?? [],
      signal_health,
      alpha_quality: {
        hit_rates,
        n_outcomes_30d:  withOutcomes.filter((p) => p.target_date >= d30).length,
        n_outcomes_120d: withOutcomes.length,
      },
    });
  } catch (err) {
    return json(500, { error: fmtErr(err) });
  }
});

// ── Signal health ──────────────────────────────────────────────────────────────
type PredRow = { target_date: string; probability_up: number; confidence?: number; direction?: string; realized_direction?: string; realized_return: number };

function computeSignalHealth(preds: PredRow[], d30: string) {
  const byDate: Record<string, { prob: number; ret: number }[]> = {};
  for (const p of preds) {
    if (!byDate[p.target_date]) byDate[p.target_date] = [];
    byDate[p.target_date].push({ prob: p.probability_up, ret: p.realized_return });
  }

  const dailyIC: { date: string; ic: number; spreadBps: number }[] = [];
  for (const [date, items] of Object.entries(byDate)) {
    if (items.length < 10) continue;
    const ic     = spearmanIC(items.map((i) => i.prob), items.map((i) => i.ret));
    const sorted = [...items].sort((a, b) => b.prob - a.prob);
    const n5     = Math.max(1, Math.floor(sorted.length / 5));
    const topRet = sorted.slice(0, n5).reduce((s, i) => s + i.ret, 0) / n5;
    const botRet = sorted.slice(-n5).reduce((s, i) => s + i.ret, 0) / n5;
    dailyIC.push({ date, ic, spreadBps: (topRet - botRet) * 10000 });
  }
  dailyIC.sort((a, b) => a.date.localeCompare(b.date));

  if (dailyIC.length === 0)
    return { ic_mean: null, ic_30d: null, ic_prior_30d: null, ic_trend: "unknown", n_ic_days: 0, decile_spread_bps: null, ic_series: [] };

  const recent      = dailyIC.filter((d) => d.date >= d30);
  const prior       = dailyIC.filter((d) => d.date < d30);
  const ic_mean     = mean(dailyIC.map((d) => d.ic));
  const ic_30d      = recent.length > 0 ? mean(recent.map((d) => d.ic)) : null;
  const ic_prior_30d = prior.length > 0 ? mean(prior.map((d) => d.ic)) : null;
  const decile_spread_bps = mean(dailyIC.slice(-30).map((d) => d.spreadBps));

  return {
    ic_mean: round(ic_mean, 4),
    ic_30d:  ic_30d !== null ? round(ic_30d, 4) : null,
    ic_prior_30d: ic_prior_30d !== null ? round(ic_prior_30d, 4) : null,
    ic_trend: ic_30d !== null && ic_prior_30d !== null
      ? (ic_30d > ic_prior_30d ? "improving" : "declining")
      : "unknown",
    n_ic_days: dailyIC.length,
    decile_spread_bps: round(decile_spread_bps, 1),
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
      (p) => p.realized_direction != null &&
        (p.confidence ?? p.probability_up) >= tier.lo &&
        (p.confidence ?? p.probability_up) < tier.hi
    );
    if (filtered.length < 5) continue;
    const correct = filtered.filter((p) => p.direction === p.realized_direction).length;
    result[tier.label] = { n: filtered.length, hit_rate: round(correct / filtered.length, 3) };
  }
  return result;
}

// ── Math ──────────────────────────────────────────────────────────────────────
function spearmanIC(probs: number[], rets: number[]): number { return pearson(rankArr(probs), rankArr(rets)); }
function rankArr(arr: number[]): number[] {
  const idx = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const r = new Array(arr.length);
  for (let i = 0; i < idx.length; i++) r[idx[i].i] = i + 1;
  return r;
}
function pearson(x: number[], y: number[]): number {
  const n = x.length, mx = mean(x), my = mean(y);
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { num += (x[i]-mx)*(y[i]-my); dx += (x[i]-mx)**2; dy += (y[i]-my)**2; }
  return dx > 0 && dy > 0 ? num / Math.sqrt(dx * dy) : 0;
}
function mean(a: number[]): number { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }

// ── Supabase helpers ──────────────────────────────────────────────────────────
function makeSB() {
  const url = Deno.env.get("SUPABASE_URL");
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !key) throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, key, { auth: { persistSession: false } });
}
type SB = ReturnType<typeof makeSB>;
type Opts = { eq?: Record<string, unknown>; gte?: Record<string, unknown>; orderDesc?: string; limit?: number };

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

// ── Utilities ─────────────────────────────────────────────────────────────────
function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status, headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
function daysAgo(n: number): string { const d = new Date(); d.setUTCDate(d.getUTCDate() - n); return d.toISOString().slice(0, 10); }
function round(v: number, n: number): number { const p = 10**n; return Math.round(v*p)/p; }
function fmtErr(e: unknown): string { if (e instanceof Error) return e.message; try { return JSON.stringify(e); } catch { return String(e); } }
