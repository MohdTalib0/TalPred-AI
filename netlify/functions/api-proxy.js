exports.handler = async (event) => {
  const baseUrl = process.env.TALPRED_API_BASE_URL;
  const apiKey = process.env.TALPRED_DASHBOARD_API_KEY || "";

  if (!baseUrl) {
    return json(500, { error: "Missing TALPRED_API_BASE_URL env var" });
  }

  const method = event.httpMethod || "GET";
  if (method !== "GET") {
    return json(405, { error: "Only GET supported" });
  }

  const path = event.queryStringParameters?.path || "";
  if (!path.startsWith("/")) {
    return json(400, { error: "Query param 'path' must start with '/'" });
  }
  if (path.includes("..")) {
    return json(400, { error: "Invalid path" });
  }

  const qs = event.queryStringParameters?.qs || "";
  const url = `${baseUrl.replace(/\/+$/, "")}${path}${qs ? `?${qs}` : ""}`;

  try {
    const headers = {};
    if (apiKey) {
      headers["x-api-key"] = apiKey;
    }

    const resp = await fetch(url, { headers });
    const contentType = resp.headers.get("content-type") || "application/json";
    const bodyText = await resp.text();

    return {
      statusCode: resp.status,
      headers: {
        "content-type": contentType,
        "cache-control": "no-store",
      },
      body: bodyText,
    };
  } catch (err) {
    return json(502, { error: `Proxy request failed: ${String(err)}` });
  }
};

function json(statusCode, obj) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json",
      "cache-control": "no-store",
    },
    body: JSON.stringify(obj),
  };
}
