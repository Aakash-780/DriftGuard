"""Persistent stores: event history (JSONL), pipeline state (JSON), labeled buffer (CSV)."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from driftguard.core.logging import get_logger
from driftguard.core.utils import atomic_write_text, to_jsonable, utc_now_iso

log = get_logger("store")


class HistoryStore:
    """Append-only event log. One JSON object per line; safe to tail and to re-load."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._events: list[dict] = []
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                self._register(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping corrupt history line %d in %s", i + 1, self.path)

    def _register(self, ev: dict) -> None:
        self._events.append(ev)
        if ev.get("event_type") == "monitoring" and ev.get("status") == "ok" and ev.get("batch_fingerprint"):
            self._seen.add(ev["batch_fingerprint"])

    def append(self, event: dict) -> dict:
        with self._lock:
            ev = to_jsonable(dict(event))
            ev.setdefault("event_id", uuid.uuid4().hex[:12])
            ev.setdefault("timestamp", utc_now_iso())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, allow_nan=False) + "\n")
            self._register(ev)
            return ev

    def has_batch(self, fingerprint: str) -> bool:
        return fingerprint in self._seen

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def recent(self, limit: int = 50, event_type: str | None = None) -> list[dict]:
        evs = [e for e in self.all() if event_type in (None, e.get("event_type"))]
        return evs[-limit:][::-1]  # newest first

    def clear(self) -> None:
        with self._lock:
            self._events, self._seen = [], set()
            if self.path.exists():
                self.path.unlink()


@dataclass
class PipelineState:
    batch_counter: int = 0
    drift_streak: int = 0
    perf_streak: int = 0
    last_retrain_batch: int | None = None
    retrain_attempts: int = 0
    promotions: int = 0
    last_window_fp: str | None = None


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = PipelineState()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.state = PipelineState(**{k: v for k, v in data.items() if k in PipelineState.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                log.warning("could not read %s; starting from a fresh state", self.path)

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(asdict(self.state), indent=2))

    def reset(self) -> None:
        self.state = PipelineState()
        if self.path.exists():
            self.path.unlink()


class BufferStore:
    """Labeled rows retained for retraining, tagged with row id, batch fingerprint and sequence."""

    META = ["row_id", "label", "batch_fp", "batch_seq"]

    def __init__(self, path: Path, feature_names, max_rows: int = 2000):
        self.path = Path(path)
        self.features = list(feature_names)
        self.max_rows = max_rows
        self.df = self._empty()
        if self.path.exists():
            try:
                df = pd.read_csv(self.path, dtype={"row_id": str, "batch_fp": str})
                if set(self.features + self.META) <= set(df.columns):
                    self.df = df
            except Exception:  # pragma: no cover
                log.warning("could not read buffer %s; starting empty", self.path)

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.features + self.META)

    def add(self, X: pd.DataFrame, y, row_ids, batch_fp: str, batch_seq: int) -> None:
        new = X[self.features].copy().reset_index(drop=True)
        new["row_id"] = [str(r) for r in row_ids]
        new["label"] = pd.Series(y).to_numpy(dtype="int64")
        new["batch_fp"] = batch_fp
        new["batch_seq"] = int(batch_seq)
        self.df = pd.concat([self.df, new], ignore_index=True) if len(self.df) else new
        if len(self.df) > self.max_rows:
            self.df = self.df.iloc[-self.max_rows:].reset_index(drop=True)
        self.df.to_csv(self.path, index=False)

    def window(self, n_batches: int) -> pd.DataFrame:
        if self.df.empty:
            return self.df.copy()
        seqs = sorted(self.df["batch_seq"].unique())[-n_batches:]
        return self.df[self.df["batch_seq"].isin(seqs)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def clear(self) -> None:
        self.df = self._empty()
        if self.path.exists():
            self.path.unlink()
