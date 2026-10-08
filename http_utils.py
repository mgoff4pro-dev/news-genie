"""Shared HTTP helper with timeouts, retries, and clear error types."""

from __future__ import annotations

import time

import requests

from . import config


class APIError(Exception):
    """An external API call failed. `kind` tells the workflow how to react."""

    def __init__(self, service: str, kind: str, detail: str = ""):
        self.service = service
        self.kind = kind  # missing_key | auth | rate_limit | timeout | network | server | bad_response
        self.detail = detail
        super().__init__(f"{service}: {kind} {detail}".strip())

    def user_message(self) -> str:
        messages = {
            "missing_key": f"{self.service} key is not set",
            "auth": f"{self.service} rejected the API key",
            "rate_limit": f"{self.service} rate limit reached",
            "timeout": f"{self.service} timed out",
            "network": f"could not reach {self.service}",
            "server": f"{self.service} is having problems",
            "bad_response": f"{self.service} returned an unexpected response",
        }
        return messages.get(self.kind, f"{self.service} error")


def request_json(
    service: str,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    retries: int | None = None,
) -> dict:
    """Make an HTTP request and return parsed JSON.

    Retries with exponential backoff on timeouts, network errors, rate limits
    and 5xx errors. Authentication errors are not retried because they will
    not fix themselves.
    """
    retries = config.HTTP_MAX_RETRIES if retries is None else retries
    headers = {"User-Agent": "NewsGenie/1.0 (course project)", **(headers or {})}
    last_error: APIError | None = None

    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(2 ** (attempt - 1), 4))  # 1s, 2s, 4s
        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            last_error = APIError(service, "timeout")
            continue
        except requests.RequestException as exc:
            last_error = APIError(service, "network", type(exc).__name__)
            continue

        if response.status_code in (401, 403):
            raise APIError(service, "auth", f"HTTP {response.status_code}")
        if response.status_code == 429:
            last_error = APIError(service, "rate_limit")
            continue
        if response.status_code >= 500:
            last_error = APIError(service, "server", f"HTTP {response.status_code}")
            continue
        if response.status_code >= 400:
            raise APIError(service, "bad_response", f"HTTP {response.status_code}")

        try:
            return response.json()
        except ValueError:
            raise APIError(service, "bad_response", "not JSON") from None

    assert last_error is not None
    raise last_error
