"""Langfuse observability layer for the asthma RAG pipeline.

Provides a configured client factory plus guarded helpers (scores, metadata,
flush) and a ``traced`` decorator so the pipeline can emit spans without ever
importing langfuse directly — and without breaking when keys are missing.

The client is constructed with explicit keys from ``Settings`` (pydantic
settings does not export ``.env`` values to ``os.environ``, so the SDK's
env-based singleton would see nothing). Spans are created through
``start_as_current_observation`` rather than the ``@observe`` decorator for
the same reason: ``@observe`` resolves its client via ``get_client()`` from
environment variables, not from our configuration.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, TypeVar

from asthma_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: Any | None = None
_initialized: bool = False
_lock = threading.Lock()

F = TypeVar("F", bound=Callable[..., Any])


def _build_client(settings: Settings) -> Any | None:
    if not settings.langfuse_tracing_enabled:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url or "https://cloud.langfuse.com",
    )


def reset_langfuse_cache() -> None:
    """Clear the cached client (used by tests to control configuration)."""
    global _client, _initialized
    with _lock:
        _client = None
        _initialized = False


def get_langfuse() -> Any | None:
    """Return the configured Langfuse client, or None when tracing is off.

    Never raises: a failing constructor is logged and treated as disabled so
    observability can never take the pipeline down.
    """
    global _client, _initialized
    if not _initialized:
        with _lock:
            if not _initialized:
                try:
                    _client = _build_client(get_settings())
                except Exception:
                    logger.warning("Langfuse client construction failed", exc_info=True)
                    _client = None
                _initialized = True
    return _client


def traced(name: str, as_type: str = "span") -> Callable[[F], F]:
    """Decorate a pipeline step to emit a Langfuse observation.

    No-ops (zero overhead, plain call-through) when tracing is disabled. The
    decorated function runs inside ``start_as_current_observation`` so nested
    ``traced`` calls form a span tree; exceptions are recorded as ERROR level
    and re-raised.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_langfuse()
            if client is None:
                return fn(*args, **kwargs)
            with client.start_as_current_observation(
                name=name, as_type=as_type
            ) as span:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    span.update(level="ERROR", status_message=str(exc))
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def score_trace(name: str, value: float | str, data_type: str = "CATEGORICAL") -> None:
    """Attach a score to the current trace; no-op when tracing is disabled."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, data_type=data_type)
    except Exception:
        logger.warning("langfuse score_trace failed for %s", name, exc_info=True)


def add_metadata(metadata: dict[str, Any]) -> None:
    """Attach metadata to the current observation; no-op when disabled."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.update_current_span(metadata=metadata)
    except Exception:
        logger.warning("langfuse add_metadata failed", exc_info=True)


def set_trace_attributes(**kwargs: Any) -> None:
    """Set trace-level attributes (e.g. session_id, tags) on the current trace."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.update_current_span(**kwargs)
    except Exception:
        logger.warning("langfuse set_trace_attributes failed", exc_info=True)


def flush_traces() -> None:
    """Block until queued traces are delivered; no-op when disabled.

    Short-lived CLI processes must call this before exit or traces are lost;
    long-running UI processes rely on the SDK's background flushing.
    """
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.warning("langfuse flush failed", exc_info=True)
