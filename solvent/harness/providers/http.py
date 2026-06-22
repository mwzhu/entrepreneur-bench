from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable


RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def urlopen_json(
    request: urllib.request.Request,
    *,
    timeout: int = 120,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST/GET JSON with conservative retry behavior for long experiment runs."""

    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"provider API error {exc.code}: {body}") from exc
            sleep(_retry_delay(exc, attempt))
        except urllib.error.URLError as exc:
            if attempt == attempts:
                raise RuntimeError(f"provider API transport error: {exc.reason}") from exc
            sleep(_retry_delay(None, attempt))
    raise RuntimeError("provider API request failed")


def _retry_delay(exc: urllib.error.HTTPError | None, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc is not None and exc.headers is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2 ** (attempt - 1)))
