# ML-02 — Deployment checkpoint

- Code: FastAPI backend and HTTP-only Streamlit frontend implemented.
- Source repository (private): https://github.com/DPEldich/household-power-forecast ; deployment branch `main`.
- Render setup: https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FDPEldich%2Fhousehold-power-forecast
- Streamlit setup: https://share.streamlit.io/ ; choose repository above, branch `main`, entrypoint `frontend/app.py`, Python 3.13.
- Local verification: 48 tests passed, including real HTTP/frontend, auth, version mismatch and causal input checks.
- Model release: `uci-hourly-1592c3f73c2fbedd`, 7 models, 4,900 test hours; bundle approximately 4.6 MB.
- Cloud configuration: `render.yaml` selects Render Free, Streamlit Cloud entrypoint is `frontend/app.py`.
- Public backend/frontend deployment: pending account connection; do not treat localhost or a source repository URL as a live cloud deployment.
- Render: no connected provider tool or `RENDER_API_KEY` currently available. Account owner must connect Render/authorize the source repository before service creation.
- Streamlit Cloud: account owner must sign in and authorize the repository; set `API_BASE_URL` to the actual live backend URL.
- Resume: verify provider connection, deploy backend, verify `/health` and `/v1/predict`, deploy frontend with actual backend URL, verify live inference for Persistence and XGBoost. Use only free compute.
