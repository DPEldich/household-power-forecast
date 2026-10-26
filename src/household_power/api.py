"""FastAPI boundary: validated requests, trusted immutable bundle, no training endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import secrets
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .serving import ForecastService

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def hourly_timestamp(value: datetime) -> datetime:
    if value.tzinfo is not None or value.minute or value.second or value.microsecond:
        raise ValueError("Use a naive local timestamp at an exact hour (YYYY-MM-DDTHH:00:00)")
    return value


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    active_power_kw: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] | None
    _hourly = field_validator("timestamp")(hourly_timestamp)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=64)
    target_timestamp: datetime
    expected_model_version: str = Field(min_length=1, max_length=100)
    history: list[Observation] | None = Field(default=None, min_length=168, max_length=336)
    _hourly = field_validator("target_timestamp")(hourly_timestamp)


class IntervalPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_timestamp: datetime
    expected_model_version: str = Field(min_length=1, max_length=100)
    history: list[Observation] | None = Field(default=None, min_length=168, max_length=336)
    _hourly = field_validator("target_timestamp")(hourly_timestamp)



def create_app(bundle_dir: Path | None = None, api_key: str | None = None) -> FastAPI:
    location = bundle_dir or Path(os.environ.get("HOUSEHOLD_POWER_BUNDLE_DIR", ROOT / "deploy/bundle"))
    required_key = os.environ.get("HOUSEHOLD_POWER_API_KEY", "") if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup if any model, preprocessing contract or checksum is missing/corrupt.
        app.state.service = ForecastService(location)
        yield

    app = FastAPI(title="Household Power Forecast API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def authorize(x_api_key: Annotated[str | None, Header()] = None):
        if required_key and not secrets.compare_digest(x_api_key or "", required_key):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health():
        service = app.state.service
        return {"status": "ok", "model_version": service.version, "models": len(service.models)}

    @app.get("/v1/dashboard", dependencies=[Depends(authorize)])
    def dashboard():
        return app.state.service.dashboard()

    @app.get("/v1/history", dependencies=[Depends(authorize)])
    def history(target_timestamp: datetime, hours: Annotated[int, Query(ge=1, le=168)] = 24):
        try:
            hourly_timestamp(target_timestamp)
            return {"model_version": app.state.service.version,
                    "history": app.state.service.history(pd.Timestamp(target_timestamp), hours)}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/predict", dependencies=[Depends(authorize)])
    def predict(request: PredictionRequest):
        service = app.state.service
        if request.expected_model_version != service.version:
            raise HTTPException(409, "Model version changed. Refresh the dashboard before predicting.")
        try:
            return service.predict(
                request.model, pd.Timestamp(request.target_timestamp),
                None if request.history is None else [item.model_dump() for item in request.history],
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/predict_interval", dependencies=[Depends(authorize)])
    def predict_interval(request: IntervalPredictionRequest):
        service = app.state.service
        if request.expected_model_version != service.version:
            raise HTTPException(409, "Model version changed. Refresh the dashboard before predicting.")
        if not getattr(service, "has_experiments", False):
            raise HTTPException(501, "Experimental interval forecasting is not available in the current bundle.")
        try:
            return service.predict_interval(
                pd.Timestamp(request.target_timestamp),
                None if request.history is None else [item.model_dump() for item in request.history],
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
