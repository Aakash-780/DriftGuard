import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from driftguard.core.config import DriftConfig  # noqa: E402
from driftguard.monitoring.evidently_adapter import EvidentlyDetector  # noqa: E402


def test_evidently_flags_shift_and_not_noise():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    same = pd.DataFrame(rng.normal(size=(150, 6)), columns=ref.columns)
    shifted = same + 2.0
    det = EvidentlyDetector(DriftConfig()).fit(ref)
    assert det.detect(same).drifted is False
    res = det.detect(shifted)
    assert res.drifted is True and res.n_drifted_features == 6 and set(res.feature_drifted) == set(ref.columns)
