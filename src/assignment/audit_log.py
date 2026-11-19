"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, deque[dict]] = defaultdict(deque)

    @staticmethod
    def _correlation_key(user_id: str, request_id: str | None) -> str:
        return f"request:{request_id}" if request_id else f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input and its start time until its output is recorded."""
        self._open[self._correlation_key(user_id, request_id)].append(
            {
                "user_id": user_id,
                "request_id": request_id,
                "input": text,
                "started_at": utc_now_iso(),
                "_started_monotonic": time.perf_counter(),
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the completed interaction and its latency."""
        key = self._correlation_key(user_id, request_id)
        pending = self._open[key].popleft() if self._open[key] else {}
        if not self._open[key]:
            self._open.pop(key)

        started = pending.get("_started_monotonic")
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 3)
            if started is not None
            else None
        )
        self.logs.append(
            {
                "request_id": request_id or pending.get("request_id"),
                "user_id": user_id,
                "input": pending.get("input"),
                "started_at": pending.get("started_at"),
                "completed_at": utc_now_iso(),
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
