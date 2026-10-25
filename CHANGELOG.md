# Changelog

## 2026-10-25 — ML-02 FastAPI + Streamlit deployment architecture

Implemented directly by Codex following the teacher's proposed architecture and Pybites reference. Streamlit is now an HTTP client only; FastAPI owns model loading, history and inference. Added bounded/validated custom-history forecasts, optional API key, explicit version mismatch responses and immutable checksum-verified serving bundle. Training/validation protocol and frozen test results remain unchanged.

Added Render Free Blueprint, separate backend/frontend dependencies, Streamlit Cloud entrypoint/secrets example, two-process local launcher and DEPLOYMENT.md. Export independently verified all seven models against all 4,900 test predictions. Full verification: 48 tests passed, including a real temporary HTTP backend and an isolated frontend process that imports no model libraries; pip check passed. Known non-failing warnings: Starlette testclient/httpx deprecation and headless notebook plots.

Cloud publish is pending Render and Streamlit Cloud account authorization; see DEPLOYMENT_STATUS.md. No service is represented as publicly live before its URLs are verified. Existing ML-01 remains complete.

## 2026-10-25 — ML-01 owner-reviewed completion

Implemented through the user-requested Antigravity exception using `gemini-3.8-flash-high`, conversation `8229badc-ea12-497f-9cdb-78e425ffceb6`. One sequential conversation; two owner refactor rounds. No public deployment or external account changes.

Delivered real UCI download and hourly aggregation, causal lag/rolling/calendar features, expanding temporal CV and chronological train/validation/test, two baselines and five ML regressors, common-mask metrics, validation and predeclared test ablations, high-load error analysis, model files, plots, predictions, executed EDA notebook, Vietnamese README/report and Streamlit historical replay with saved-model inference.

Owner review corrected missing baseline lag integration, app package imports hidden by pytest, baseline improvement percentages, misleading calendar coverage labels, unsupported UI/report interpretations, report values inconsistent with CSV, false notebook execution markers, calendar-window plots and inconsistent feature order in full ablation. Reports now regenerate from current CSV/JSON using `scripts/generate_reports.py`; EDA generation executes real notebook cells and captures their outputs.

Independent verification:

- `.venv/Scripts/python.exe -m pytest -q`: 32 passed in 17.13 seconds; two expected FigureCanvasAgg warnings from executing notebook plotting cells headlessly.
- `.venv/Scripts/python.exe -m pip check`: no broken requirements.
- All seven saved model predictions matched all 4,900 stored test rows, checked against observed targets; independently recalculated MAE/RMSE and sample counts matched CSV.
- Full ablation MAE/RMSE/R2 matched primary model for validation and test.
- Reviewed source, scope, dependency versions, output plots, metadata and report tables. Git whitespace check passed with CRLF and intentional Markdown line breaks allowed.
- Streamlit started locally at `http://127.0.0.1:8501`; health endpoint and HTTP response verified. Start again with `start_demo.bat` if the current process stops.

Validation-selected XGBoost: test MAE 0.3242076 kW, RMSE 0.4665532 kW, R2 0.5697772; MAE improvement 16.70% vs Persistence and 36.42% vs Seasonal Naive. Test covers 4,900/5,164 calendar hours (216 invalid targets and 48 additional unavailable-baseline hours excluded). Hyperparameters were not tuned on test; engineering reruns retained the same selection protocol and configs.

Boundaries: one French household (2006–2010), one-step hourly forecasts with sequential actual observations; no weather inputs, multi-step forecast, or measured electricity-bill savings. Raw data, environment and binary models remain local ignored files and must be downloaded/regenerated when transferring only Git-tracked source. Restart Streamlit after regenerating artifacts to refresh caches. No downstream task remains queued.
