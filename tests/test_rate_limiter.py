from server.services.rate_limiter import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit_then_allows_after_clear():
    limiter = InMemoryRateLimiter()
    key = "u:test"

    assert limiter.allow(key, limit=2, window_sec=60) is True
    assert limiter.allow(key, limit=2, window_sec=60) is True
    assert limiter.allow(key, limit=2, window_sec=60) is False

    limiter.clear()
    assert limiter.allow(key, limit=2, window_sec=60) is True
