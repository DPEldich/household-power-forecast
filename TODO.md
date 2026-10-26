# ML electricity forecasting

| Task | Executor | Model | Mode | Write scope | Status |
|---|---|---|---|---|---|
| ML-01 | Antigravity | gemini-3.8-flash-high | sequential | Entire project except owner closeout files | reviewed / complete |
| ML-02 | Codex | current | sequential | FastAPI, HTTP frontend, deploy bundle/config/docs/tests | code reviewed; cloud publish waiting for account access |
| ML-03 | Antigravity + Codex review | gemini-3.8-flash-high | sequential | See ML03_BRIEF.md; isolated experiments + optional serving/UI | reviewed / complete; baseline retained, experimental P10/P50/P90 available locally |

Acceptance: real UCI hourly pipeline, seven models/baselines, temporal validation, real held-out metrics, ablation, functioning Streamlit demo, Vietnamese documentation/report, reproducible commands and meaningful tests. Owner review required before closeout.

ML-03 owner closeout 2026-10-26 (Asia/Saigon): passed after two Antigravity refactor loops and independent Codex review. Independent full pytest: 80 passed, 3 known warnings; pip check passed. All-row serving export verified seven original models plus point/P10/P50/P90 estimators over 4,900 benchmark hours. Original 16 baseline artifact SHA256 hashes unchanged. Candidate rejected by validation high-load criterion; point model remains baseline. Experimental intervals have 90.49% overall vs 38.05% high-load coverage on reused test data, not calibrated coverage guarantees. See ML03_REVIEW.md and ML03_RESULTS.md. No 24h forecast, new dataset or cloud publication was included; ML-02 account blocker remains unchanged.

Owner closeout 2026-10-25 (Asia/Saigon): passed after two refactor rounds. Independent pytest: 32 passed (two expected headless matplotlib warnings); pip check passed. All seven saved models independently matched all 4,900 test predictions and metrics; full ablation matched primary winner. Real UCI data downloaded, executed EDA notebook and CSV-bound Vietnamese reports reviewed. Local Streamlit demo: http://127.0.0.1:8501. See CHANGELOG.md for evidence and boundaries.
