"""Tests: capped downloads must not satisfy later uncapped requests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tiny_llm.data import (
    _cached_download_usable,
    _download_file,
    _download_meta_path,
    _write_download_meta,
)


def test_capped_cache_rejected_when_uncapped_requested(tmp_path: Path):
    dest = tmp_path / "tinystories_train.txt"
    dest.write_bytes(b"x" * 1000)
    _write_download_meta(
        dest,
        url="https://example.test/train.txt",
        nbytes=1000,
        max_bytes=1000,
        complete=False,
    )
    assert _cached_download_usable(dest, max_bytes=500) is True
    assert _cached_download_usable(dest, max_bytes=1000) is True
    assert _cached_download_usable(dest, max_bytes=2000) is False
    assert _cached_download_usable(dest, max_bytes=None) is False


def test_complete_cache_accepted_for_uncapped(tmp_path: Path):
    dest = tmp_path / "tinystories_train.txt"
    dest.write_bytes(b"full-corpus")
    _write_download_meta(
        dest,
        url="https://example.test/train.txt",
        nbytes=len(b"full-corpus"),
        max_bytes=None,
        complete=True,
    )
    assert _cached_download_usable(dest, max_bytes=None) is True
    assert _cached_download_usable(dest, max_bytes=4) is True


def test_legacy_cache_without_sidecar_not_used_uncapped(tmp_path: Path):
    dest = tmp_path / "tinystories_train.txt"
    dest.write_bytes(b"x" * 5000)
    assert not _download_meta_path(dest).exists()
    assert _cached_download_usable(dest, max_bytes=None) is False
    assert _cached_download_usable(dest, max_bytes=1000) is True


def test_download_writes_incomplete_meta_when_capped(tmp_path: Path):
    dest = tmp_path / "blob.txt"
    payload = b"abcdefghijklmnopqrstuvwxyz"  # 26 bytes

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[payload])

    with patch("tiny_llm.data.requests.get", return_value=mock_resp):
        _download_file("https://example.test/blob", dest, max_bytes=10)

    assert dest.read_bytes() == payload[:10]
    meta = _download_meta_path(dest).read_text(encoding="utf-8")
    assert '"complete": false' in meta
    assert _cached_download_usable(dest, max_bytes=None) is False


def test_download_writes_complete_meta_on_eof(tmp_path: Path):
    dest = tmp_path / "blob.txt"
    payload = b"short"

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[payload])

    with patch("tiny_llm.data.requests.get", return_value=mock_resp):
        _download_file("https://example.test/blob", dest, max_bytes=1000)

    assert dest.read_bytes() == payload
    meta = _download_meta_path(dest).read_text(encoding="utf-8")
    assert '"complete": true' in meta
    assert _cached_download_usable(dest, max_bytes=None) is True
