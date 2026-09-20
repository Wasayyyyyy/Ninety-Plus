"""
Unit tests for the DiskCache implementation.
"""

import time
from pathlib import Path
import pytest
from src.data_layer.cache import DiskCache


def test_cache_set_and_get(tmp_path: Path):
    cache = DiskCache(cache_dir=tmp_path, default_ttl_seconds=10)
    cache.set("test_key", {"a": 1, "b": "football"})

    assert cache.has("test_key")
    val = cache.get("test_key")
    assert val == {"a": 1, "b": "football"}


def test_cache_expiration(tmp_path: Path):
    cache = DiskCache(cache_dir=tmp_path, default_ttl_seconds=1)
    cache.set("short_lived", {"hello": "world"}, ttl_seconds=1)

    assert cache.get("short_lived") is not None
    time.sleep(1.2)
    # Should now be expired and return None
    assert cache.get("short_lived") is None
    assert not cache.has("short_lived")


def test_cache_delete_and_clear(tmp_path: Path):
    cache = DiskCache(cache_dir=tmp_path, default_ttl_seconds=60)
    cache.set("k1", 100)
    cache.set("k2", 200)

    assert cache.get("k1") == 100
    cache.delete("k1")
    assert cache.get("k1") is None
    assert cache.get("k2") == 200

    cache.clear()
    assert cache.get("k2") is None
