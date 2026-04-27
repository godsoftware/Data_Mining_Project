"""Calibration helpers."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrated_classifier(estimator, method: str = "isotonic", cv: int = 3):
    """Create a calibrated classifier wrapper."""

    return CalibratedClassifierCV(estimator, method=method, cv=cv)

