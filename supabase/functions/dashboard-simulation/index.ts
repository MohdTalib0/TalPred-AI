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

    const [simRunRows, tradeRows] = await Promise.all([
      selectRows(sb, "simulation_runs",
        "run_id,created_at,start_date,end_date,strategy_name,status,result_metrics,starting_capital,model_version,min_confidence_trade,max_position,transaction_cost_bps,slippage_bps",
        { eq: { status: "completed" }, orderDesc: "created_at", limit: 20 }),
      selectRows(sb, "paper_trades",
        "run_id,date,symbol,weight,position_qty,entry_price,exit_price,daily_pnl,transaction_cost,slippage_cost",
        { orderDesc: "date", limit: 200 }),
    ]);

    const runs = (simRunRows ?? []).map((r) => ({
      run_id:         r.run_id,
      created_at:     r.created_at,
      start_date:     r.start_date,
      end_date:       r.end_date,
      strategy_name:  r.strategy_name ?? "legacy",
      model_version:  r.model_version,
      starting_capital: r.starting_capital,
      min_confidence_trade: r.min_confidence_trade,
      max_position:   r.max_position,
      transaction_cost_bps: r.transaction_cost_bps,
      slippage_bps:   r.slippage_bps,
      metrics:        r.result_metrics ?? {},
    }));

    const latestMetrics = runs[0]?.metrics ?? {};

    // Aggregate trade stats across all returned trades
    const trades = tradeRows ?? [];
    const pnlValues = trades.map((t) => t.daily_pnl ?? 0);
    const totalPnl  = pnlValues.reduce((s, v) => s + v, 0);
    const winners   = pnlValues.filter((v) => v > 0).length;
    const losers    = pnlValues.filter((v) => v < 0).length;

    return json(200, {
      runs,
      latest_trades:      trades,
      latest_run_metrics: latestMetrics,
      trade_stats: {
        total_pnl:  round(totalPnl, 2),
        n_trades:   trades.length,
        n_winners:  winners,
        n_losers:   losers,
        win_rate:   trades.length > 0 ? round(winners / trades.length, 3) : null,
      },
    });
  } catch (err) {
    return json(500, { error: fmtErr(err) });
  }
});

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
function round(v: number, n: number): number { const p = 10**n; return Math.round(v*p)/p; }
function fmtErr(e: unknown): string { if (e instanceof Error) return e.message; try { return JSON.stringify(e); } catch { return String(e); } }
