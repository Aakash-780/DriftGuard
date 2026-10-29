from __future__ import annotations

import pytest

from driftguard.core.config import DataConfig, Settings
from driftguard.core.tracking import Tracker
from driftguard.data.loader import make_splits
from driftguard.pipelines.pipeline import DriftGuardPipeline


@pytest.fixture
def settings(tmp_path):
    return Settings(artifacts_dir=tmp_path / "artifacts", data=DataConfig(seed=0), mlflow_mode="false")


@pytest.fixture
def splits():
    return make_splits(DataConfig(seed=0))


@pytest.fixture
def pipeline(settings):
    p = DriftGuardPipeline(settings, tracker=Tracker.disabled())
    p.ensure_baseline()
    return p
