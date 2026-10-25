import pytest
import requests
from household_power.api_client import ApiError, request_json


def test_connection_timeout_has_safe_actionable_message(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("do not expose this internal message")
    monkeypatch.setattr(requests, "request", timeout)
    with pytest.raises(ApiError, match="Không kết nối được FastAPI") as error:
        request_json("https://example.org", "/v1/dashboard", "private-key")
    assert "private-key" not in str(error.value)


def test_html_cold_start_page_is_not_treated_as_data(monkeypatch):
    response = requests.Response()
    response.status_code = 503
    response._content = b"<html>Starting</html>"
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: response)
    with pytest.raises(ApiError, match="HTTP 503"):
        request_json("https://example.org", "/v1/dashboard")
