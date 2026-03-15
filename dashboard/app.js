async function fetchProxy(path, qs = "") {
  const url = `/api/proxy?path=${encodeURIComponent(path)}${qs ? `&qs=${encodeURIComponent(qs)}` : ""}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`${path} -> HTTP ${resp.status}`);
  }
  return resp.json();
}

async function fetchWorkflows() {
  const resp = await fetch("/api/workflows");
  if (!resp.ok) {
    throw new Error(`workflows -> HTTP ${resp.status}`);
  }
  return resp.json();
}

function setText(id, value, cls = "") {
  const el = document.getElementById(id);
  el.textContent = value;
  el.className = cls;
}

function renderAlerts(alerts) {
  const container = document.getElementById("alerts-list");
  container.innerHTML = "";
  if (!alerts || alerts.length === 0) {
    container.innerHTML = `<div class="list-item ok">No active alerts</div>`;
    return;
  }
  for (const a of alerts.slice(0, 10)) {
    const cls = a.level === "critical" ? "bad" : "warn";
    const div = document.createElement("div");
    div.className = `list-item ${cls}`;
    div.innerHTML = `<strong>${a.level.toUpperCase()}</strong> ${a.check}<br><small>${a.detail}</small>`;
    container.appendChild(div);
  }
}

function renderWorkflows(data) {
  const container = document.getElementById("workflow-list");
  container.innerHTML = "";
  if (!data.available) {
    container.innerHTML = `<div class="list-item warn">${data.reason || "Workflow data unavailable"}</div>`;
    return;
  }
  for (const run of data.runs) {
    const cls =
      run.conclusion === "success"
        ? "ok"
        : run.conclusion === "failure" || run.conclusion === "error"
          ? "bad"
          : "warn";
    const div = document.createElement("div");
    div.className = `list-item ${cls}`;
    const link = run.html_url ? `<a href="${run.html_url}" target="_blank" rel="noreferrer">open</a>` : "";
    div.innerHTML =
      `<strong>${run.workflow.replace(".github/workflows/", "")}</strong><br>` +
      `<small>status=${run.status} conclusion=${run.conclusion} ${link}</small>`;
    container.appendChild(div);
  }
}

function renderPredictions(rows) {
  const tbody = document.querySelector("#predictions-table tbody");
  tbody.innerHTML = "";
  for (const p of rows || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${p.symbol}</td>
      <td>${p.target_date}</td>
      <td>${p.direction}</td>
      <td>${Number(p.probability_up).toFixed(3)}</td>
      <td>${Number(p.confidence).toFixed(3)}</td>
      <td>${p.model_version || "-"}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function refreshDashboard() {
  try {
    const [health, model, monitoring, predictions, workflows] = await Promise.all([
      fetchProxy("/health"),
      fetchProxy("/model/info").catch(() => null),
      fetchProxy("/monitoring/status"),
      fetchProxy("/predictions", "limit=25&min_confidence=0.55"),
      fetchWorkflows(),
    ]);

    setText(
      "health-status",
      health?.status || "unknown",
      health?.status === "healthy" ? "ok" : "warn",
    );
    setText("health-meta", `cache=${health?.cache_available ? "yes" : "no"} v${health?.version || "-"}`);

    if (model) {
      setText("model-version", model.model_version || "-", "ok");
      setText("model-meta", `${model.algorithm || "unknown"} | ${model.status || "-"}`);
    } else {
      setText("model-version", "unavailable", "warn");
      setText("model-meta", "No production model info");
    }

    const overall = monitoring?.overall_status || "unknown";
    const overallCls = overall === "healthy" ? "ok" : overall === "degraded" ? "warn" : "bad";
    setText("monitoring-overall", overall, overallCls);
    setText("monitoring-alerts", `${monitoring?.total_alerts ?? 0} alert(s)`);

    setText("prediction-count", `${predictions?.length ?? 0}`, "ok");
    const latestDate = predictions?.[0]?.target_date || "-";
    setText("prediction-meta", `latest target=${latestDate}`);

    renderAlerts(monitoring?.alert_details || []);
    renderWorkflows(workflows);
    renderPredictions(predictions || []);

    const ts = new Date().toLocaleString();
    document.getElementById("last-updated").textContent = `Last updated: ${ts}`;
  } catch (err) {
    document.getElementById("last-updated").textContent = `Refresh failed: ${String(err)}`;
  }
}

document.getElementById("refresh-btn").addEventListener("click", refreshDashboard);
refreshDashboard();
