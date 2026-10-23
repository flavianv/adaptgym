# Deploy on Render

The root [render.yaml](../render.yaml) defines a Python service named `adaptgym`. It installs the package and starts the dashboard on `0.0.0.0:$PORT`.

1. Push the repository to GitHub and connect it through Render's **New → Blueprint** flow.
2. Select `main` and the root `render.yaml`.
3. Supply `OPENAI_API_KEY` as a server-side secret. Set `OPENAI_MODEL` to a model available to your account.
4. Open the deployed URL. The saved product-selection example loads without making API calls.

The dashboard has no login. Starting a live run uses the server's API key and makes up to four calls per shopper turn. The key is never sent to the browser. `/healthz` returns `{"status":"ok"}`. One live run is allowed at a time; live results are held in memory, and a restart restores the bundled example.

Render's hostname is accepted automatically. For a custom domain, set `ADAPTGYM_PUBLIC_URL` to its exact origin, for example `https://demo.example.com`. Cross-origin POSTs are blocked.

For the local equivalent:

```bash
python -m pip install .
python scripts/debug_session_scenarios.py --serve --no-browser --host 127.0.0.1 --port 8768
```

No frontend build or database is required.
