"""Local, file-based model registry (source of truth for serving).

Layout::

    <root>/current.json          {"version": N}   (atomically replaced on promotion)
    <root>/v0001/model.joblib
    <root>/v0001/meta.json

Versions are integers; no user-supplied path is ever used. NOTE: joblib/pickle files must only be
loaded from a trusted directory -- this registry never accepts uploads or paths from API callers.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

from driftguard.core.utils import atomic_write_text, to_jsonable, utc_now_iso

_VERSION_DIR = re.compile(r"^v(\d{4,})$")


@dataclass
class ModelRecord:
    version: int
    status: str  # candidate | production | archived | rejected
    created_at: str
    parent_version: int | None = None
    trigger_reason: str | None = None
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reference_metrics: dict = field(default_factory=dict)  # performance baseline for this model
    train_rows: int = 0
    data_fingerprint: str = ""
    mlflow_run_id: str | None = None
    decision: dict | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def _dir(self, version: int) -> Path:
        return self.root / f"v{int(version):04d}"

    def _versions(self) -> list[int]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.iterdir():
            m = _VERSION_DIR.match(p.name)
            if m and p.is_dir():
                out.append(int(m.group(1)))
        return sorted(out)

    # ------------------------------------------------------------------ read
    def current_version(self) -> int | None:
        f = self.root / "current.json"
        if not f.exists():
            return None
        try:
            return int(json.loads(f.read_text(encoding="utf-8"))["version"])
        except (ValueError, KeyError, json.JSONDecodeError):
            return None

    def get(self, version: int) -> ModelRecord:
        meta = json.loads((self._dir(version) / "meta.json").read_text(encoding="utf-8"))
        return ModelRecord(**meta)

    def list(self) -> list[ModelRecord]:
        return [self.get(v) for v in self._versions()]

    def load(self, version: int):
        return joblib.load(self._dir(version) / "model.joblib")

    def load_current(self):
        v = self.current_version()
        if v is None:
            raise FileNotFoundError("no model has been promoted yet")
        return self.load(v), self.get(v)

    # ------------------------------------------------------------------ write
    def _write_meta(self, rec: ModelRecord) -> None:
        atomic_write_text(self._dir(rec.version) / "meta.json", json.dumps(rec.to_dict(), indent=2))

    def register(self, model, **fields: Any) -> ModelRecord:
        """Store a model as a new ``candidate`` version (does not change what is served)."""
        with self._lock:
            versions = self._versions()
            version = (versions[-1] + 1) if versions else 1
            self._dir(version).mkdir(parents=True, exist_ok=False)
            joblib.dump(model, self._dir(version) / "model.joblib", compress=3)
            rec = ModelRecord(
                version=version, status="candidate", created_at=utc_now_iso(),
                parent_version=self.current_version(), **fields,
            )
            self._write_meta(rec)
            return rec

    def update(self, version: int, **fields: Any) -> ModelRecord:
        with self._lock:
            rec = self.get(version)
            for k, v in fields.items():
                if not hasattr(rec, k):
                    raise AttributeError(k)
                setattr(rec, k, v)
            self._write_meta(rec)
            return rec

    def promote(self, version: int) -> ModelRecord:
        with self._lock:
            if version not in self._versions():
                raise KeyError(f"unknown version {version}")
            prev = self.current_version()
            if prev is not None and prev != version:
                self.update(prev, status="archived")
            rec = self.update(version, status="production")
            atomic_write_text(self.root / "current.json", json.dumps({"version": version}))
            return rec

    def reject(self, version: int, reason: str = "") -> ModelRecord:
        with self._lock:
            return self.update(version, status="rejected", notes=reason)
