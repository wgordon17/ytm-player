"""Tests for YTMusicService.get_multi_seed_radio."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from ytm_player.services.ytmusic import YTMusicService


@pytest.fixture
def svc():
    """Construct YTMusicService bypassing __init__."""
    s = YTMusicService.__new__(YTMusicService)
    s._auth_path = None
    s._auth_manager = None
    s._user = None
    s._consecutive_api_failures = 0
    s._order_lock = asyncio.Lock()
    s._ytm = None
    return s


def _tracks(video_ids: list[str]) -> list[dict]:
    return [{"videoId": vid, "title": f"Track {vid}"} for vid in video_ids]


class TestGetMultiSeedRadio:
    async def test_deduplication_across_seeds(self, svc):
        """Tracks returned by multiple seeds are deduplicated by video_id."""
        seed_results = {
            "v1": _tracks(["a", "b", "c"]),
            "v2": _tracks(["b", "c", "d"]),  # b and c are duplicates
            "v3": _tracks(["e"]),
        }

        async def fake_get_radio(video_id):
            return seed_results.get(video_id, [])

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            result = await svc.get_multi_seed_radio(["v1", "v2", "v3"], limit=25)

        result_ids = {t["videoId"] for t in result}
        assert result_ids == {"a", "b", "c", "d", "e"}

    async def test_early_stopping_at_pool_target(self, svc):
        """Stops calling get_radio once 50 unique tracks are collected."""
        # 25 unique tracks per seed; 2 seeds should reach the 50-track target
        seed1 = _tracks([f"s1_{i}" for i in range(25)])
        seed2 = _tracks([f"s2_{i}" for i in range(25)])
        seed3 = _tracks([f"s3_{i}" for i in range(25)])

        call_count = 0

        async def fake_get_radio(video_id):
            nonlocal call_count
            call_count += 1
            if video_id == "v1":
                return seed1
            if video_id == "v2":
                return seed2
            return seed3

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            await svc.get_multi_seed_radio(["v1", "v2", "v3"], limit=25)

        # Should stop after v2 (50 tracks reached) — v3 never called
        assert call_count == 2

    async def test_individual_seed_failure_does_not_abort(self, svc):
        """A failure on one seed is skipped; remaining seeds still contribute."""

        async def fake_get_radio(video_id):
            if video_id == "bad":
                raise RuntimeError("API error")
            return _tracks([f"{video_id}_track"])

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            result = await svc.get_multi_seed_radio(["bad", "good"], limit=25)

        result_ids = {t["videoId"] for t in result}
        assert "good_track" in result_ids

    async def test_result_trimmed_to_limit(self, svc):
        """Result is trimmed to the requested limit."""
        many_tracks = _tracks([f"v{i}" for i in range(40)])

        async def fake_get_radio(video_id):
            return many_tracks

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            result = await svc.get_multi_seed_radio(["seed1"], limit=10)

        assert len(result) == 10

    async def test_max_3_seeds_enforced(self, svc):
        """Only the first 3 seeds are used even if more are provided."""
        call_log: list[str] = []

        async def fake_get_radio(video_id):
            call_log.append(video_id)
            return _tracks([f"{video_id}_t"])

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            await svc.get_multi_seed_radio(["s1", "s2", "s3", "s4", "s5"], limit=25)

        assert "s4" not in call_log
        assert "s5" not in call_log
        assert call_log == ["s1", "s2", "s3"]

    async def test_empty_seeds_returns_empty(self, svc):
        """Empty seed list returns empty result without calling get_radio."""
        call_count = 0

        async def fake_get_radio(video_id):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            result = await svc.get_multi_seed_radio([], limit=25)

        assert result == []
        assert call_count == 0

    async def test_all_seeds_fail_returns_empty(self, svc):
        """If every seed fails, returns empty list."""

        async def fake_get_radio(video_id):
            raise RuntimeError("failure")

        with patch.object(svc, "get_radio", side_effect=fake_get_radio):
            result = await svc.get_multi_seed_radio(["s1", "s2", "s3"], limit=25)

        assert result == []
