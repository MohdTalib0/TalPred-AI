import { createClient } from "npm:@supabase/supabase-js@2.49.1";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  try {
    const sb  = makeSB();
    const d60 = daysAgo(60);
    const d252 = daysAgo(365); // ~252 trading days

    const [spyRows, vixRow, spyYearRows] = await Promise.all([
      // SPY 60-day OHLCV for sparkline + drawdown
      selectRows(sb, "market_bars_daily",
        "date,open,high,low,close,volume",
        { eq: { symbol: "SPY" }, gte: { date: d60 }, orderDesc: "date", limit: 70 }),
      // Latest VIX, regime, SP500 momentum from features_snapshot
      selectOne(sb, "features_snapshot",
        "vix_level,sp500_momentum_200d,regime_label,target_session_date",
        { orderDesc: "target_session_date" }),
      // SPY 1-year bars for 52w high/low
      selectRows(sb, "market_bars_daily",
        "date,close,high,low",
        { eq: { symbol: "SPY" }, gte: { date: d252 }, orderDesc: "date", limit: 300 }),
    ]);

    const spySorted = [...(spyRows ?? [])].sort((a, b) => a.date.localeCompare(b.date));
    const closes    = spySorted.map((r) => r.close);
    const { drawdown, peak } = computeDrawdown(closes);

    // 1-year stats
    const yearCloses = [...(spyYearRows ?? [])].sort((a, b) => a.date.localeCompare(b.date)).map((r) => r.close);
    const high52w = yearCloses.length ? Math.max(...yearCloses) : null;
    const low52w  = yearCloses.length ? Math.min(...yearCloses) : null;
    const lastClose = closes[closes.length - 1] ?? null;
    const pctFrom52wHigh = high52w && lastClose ? round((lastClose - high52w) / high52w * 100, 2) : null;

    // Day-over-day return
    const spy1dReturn = closes.length >= 2
      ? round((closes[closes.length - 1] - closes[closes.length - 2]) / closes[closes.length - 2] * 100, 3)
      : null;

    // Volume 5d average
    const vols5 = spySorted.slice(-5).map((r) => r.volume ?? 0);
    const avgVol5d = vols5.length ? round(vols5.reduce((s, v) => s + v, 0) / vols5.length, 0) : null;

    return json(200, {
      market_context: {
        vix:                  vixRow?.vix_level        ? round(vixRow.vix_level, 1)          : null,
        vix_label:            vixLabel(vixRow?.vix_level),
        regime:               vixRow?.regime_label     ?? null,
        sp500_momentum_200d:  vixRow?.sp500_momentum_200d ? round(vixRow.sp500_momentum_200d * 100, 2) : null,
        spy_drawdown_60d:     drawdown,
        spy_peak_60d:         peak ? round(peak, 2) : null,
        spy_last_close:       lastClose ? round(lastClose, 2) : null,
        spy_1d_return_pct:    spy1dReturn,
        spy_52w_high:         high52w ? round(high52w, 2) : null,
        spy_52w_low:          low52w  ? round(low52w, 2)  : null,
        pct_from_52w_high:    pctFrom52wHigh,
        spy_avg_volume_5d:    avgVol5d,
        as_of_date:           vixRow?.target_session_date ?? null,
      },
      spy_ohlcv: spySorted.map((r) => ({
        date:   r.date,
        open:   round(r.open, 2),
        high:   round(r.high, 2),
        low:    round(r.low, 2),
        close:  round(r.close, 2),
        volume: r.volume ?? null,
      })),
    });
  } catch (err) {
    return json(500, { error: fmtErr(err) });
  }
});

function vixLabel(vix: number | null): string {
  if (vix == null) return "unknown";
  if (vix >= 40)   return "crisis";
  if (vix >= 30)   return "high_stress";
  if (vix >= 20)   return "elevated";
  if (vix >= 12)   return "normal";
  return "low";
}

function computeDrawdown(closes: number[]) {
  if (!closes.length) return { drawdown: null, peak: null };
  let peak = closes[0], dd = 0;
  for (const c of closes) { if (c > peak) peak = c; const d = (c - peak) / peak; if (d < dd) dd = d; }
  return { drawdown: round(dd, 4), peak };
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

async function selectRows(sb: SB, table: string, cols: string, opts: Opts = {}): Promise<any[]> {
  let q = sb.from(table).select(cols);
  if (opts.eq)        for (const [k, v] of Object.entries(opts.eq))  q = q.eq(k, v);
  if (opts.gte)       for (const [k, v] of Object.entries(opts.gte)) q = q.gte(k, v);
  if (opts.orderDesc)  q = q.order(opts.orderDesc, { ascending: false });
  if (opts.limit)      q = q.limit(opts.limit);
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
function round(v: number, n: number): number { const p = 10**n; return Math.round(v*p)/p; }
function fmtErr(e: unknown): string { if (e instanceof Error) return e.message; try { return JSON.stringify(e); } catch { return String(e); } }
