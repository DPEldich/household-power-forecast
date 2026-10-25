"""HTTP-only frontend client: deliberately imports no model or training libraries."""

from __future__ import annotations

import requests


class ApiError(RuntimeError):
    pass


def request_json(base_url: str, path: str, api_key: str = "", *, body: dict | None = None, params=None):
    if not base_url.startswith(("http://", "https://")):
        raise ApiError("API_BASE_URL phải bắt đầu bằng http:// hoặc https://")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = requests.request(
            "GET" if body is None else "POST", base_url.rstrip("/") + path,
            headers=headers, json=body, params=params, timeout=(10, 90),
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            detail = "Backend chưa sẵn sàng; dịch vụ miễn phí có thể đang khởi động."
        raise ApiError(f"API trả HTTP {response.status_code}: {detail}") from exc
    except (requests.RequestException, ValueError) as exc:
        raise ApiError("Không kết nối được FastAPI. Kiểm tra API_BASE_URL và thử lại sau khi backend khởi động.") from exc
