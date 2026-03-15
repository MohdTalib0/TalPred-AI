exports.handler = async () => {
  const token = process.env.GITHUB_TOKEN_DASHBOARD || "";
  const owner = process.env.GITHUB_REPO_OWNER || "";
  const repo = process.env.GITHUB_REPO_NAME || "";

  if (!token || !owner || !repo) {
    return json(200, {
      available: false,
      reason: "Missing one of GITHUB_TOKEN_DASHBOARD, GITHUB_REPO_OWNER, GITHUB_REPO_NAME",
      runs: [],
    });
  }

  const workflows = [
    ".github/workflows/daily-pipeline.yml",
    ".github/workflows/monthly-archive.yml",
    ".github/workflows/news-sentiment.yml",
  ];

  try {
    const runs = [];
    for (const wf of workflows) {
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
    return json(200, { available: true, runs });
  } catch (err) {
    return json(500, { available: false, error: String(err), runs: [] });
  }
};

function json(statusCode, body) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json",
      "cache-control": "no-store",
    },
    body: JSON.stringify(body),
  };
}
