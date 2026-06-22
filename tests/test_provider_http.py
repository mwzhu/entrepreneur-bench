from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from solvent.harness.providers.http import urlopen_json


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.body


def test_urlopen_json_retries_429_with_retry_after(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "1.25"},
                BytesIO(b'{"error":"slow down"}'),
            )
        return _Response(b'{"ok":true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = urlopen_json(Request("https://example.test"), sleep=sleeps.append)

    assert result == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [1.25]


def test_urlopen_json_fails_after_transport_retries(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        raise URLError("temporary dns failure")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="transport error"):
        urlopen_json(Request("https://example.test"), max_attempts=2, sleep=lambda _: None)

    assert len(calls) == 2
