"""
Earth Engine rate limiting + retry helpers for Restricted Mode.
"""
import os
import time
import threading

# Minimum gap between EE getInfo / network calls (seconds)
EE_MIN_INTERVAL_SEC = float(os.getenv("EE_MIN_INTERVAL_SEC", "1.5"))
EE_MAX_RETRIES = int(os.getenv("EE_MAX_RETRIES", "6"))
EE_BACKOFF_BASE_SEC = float(os.getenv("EE_BACKOFF_BASE_SEC", "3"))

_lock = threading.Lock()
_last_call_at = 0.0


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "too many requests",
            "concurrency",
            "rate limit",
            "quota",
            "429",
            "user memory limit exceeded",
        )
    )


def ee_throttle(min_interval: float | None = None):
    """Wait so consecutive EE calls are spaced out."""
    global _last_call_at
    gap = EE_MIN_INTERVAL_SEC if min_interval is None else min_interval
    with _lock:
        now = time.time()
        wait = gap - (now - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.time()


def ee_get_info(ee_obj, label: str = "ee"):
    """
    Call .getInfo() with global throttle + exponential backoff on rate limits.
    """
    global _last_call_at
    last_err = None

    for attempt in range(1, EE_MAX_RETRIES + 1):
        ee_throttle()
        try:
            result = ee_obj.getInfo()
            return result
        except Exception as e:
            last_err = e
            with _lock:
                _last_call_at = time.time()

            if _is_rate_limit_error(e) and attempt < EE_MAX_RETRIES:
                backoff = EE_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                print(
                    f"⏳ EE rate limit on {label} "
                    f"(attempt {attempt}/{EE_MAX_RETRIES}) — sleep {backoff:.0f}s",
                    flush=True,
                )
                time.sleep(backoff)
                continue
            raise

    raise last_err


def sleep_between_windows(seconds: float | None = None):
    """Extra pause between half-month pest windows / plots."""
    pause = float(os.getenv("EE_WINDOW_PAUSE_SEC", "2")) if seconds is None else seconds
    if pause > 0:
        time.sleep(pause)
