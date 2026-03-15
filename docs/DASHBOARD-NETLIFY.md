# Netlify Ops Dashboard Setup

This project now includes a React-based ops dashboard powered by a Supabase Edge Function:

- React app: `dashboard-react/`
- Supabase function: `supabase/functions/dashboard-data/`
- Netlify config: `netlify.toml`

## What it shows

- API health (`/health`)
- Production model info (`/model/info`)
- Monitoring overall status + alerts (`/monitoring/status`)
- Latest predictions (`/predictions`)
- Optional GitHub workflow run status (daily, archive, news CI)

## Supabase Edge Function Setup

Deploy the function:

```bash
supabase functions deploy dashboard-data --no-verify-jwt
```

Set function secrets:

```bash
supabase secrets set GITHUB_TOKEN_DASHBOARD=... GITHUB_REPO_OWNER=... GITHUB_REPO_NAME=...
```

`GITHUB_*` secrets are optional; without them workflow panel will show unavailable.

## Netlify Environment Variables

Set these in Netlify Site Settings -> Environment variables:

- `VITE_DASHBOARD_EDGE_URL` (example: `https://<project-ref>.supabase.co/functions/v1/dashboard-data`)

If GitHub secrets are missing in Supabase, dashboard still works and workflow panel shows "unavailable".

## Deploy

1. Connect this repo in Netlify.
2. Build settings are auto-read from `netlify.toml`:
   - command: `npm --prefix dashboard-react install && npm --prefix dashboard-react run build`
   - publish: `dashboard-react/dist`
3. Deploy.

## Local development

Set local env with your deployed edge function URL (PowerShell):

```bash
$env:VITE_DASHBOARD_EDGE_URL="https://<project-ref>.supabase.co/functions/v1/dashboard-data"
npx netlify dev --port 8890
```

Open: `http://localhost:8890`

## Notes

- Dashboard fetches one consolidated payload from Supabase Edge Function.
- If you keep old Netlify functions in repo, they are no longer required for the dashboard.
