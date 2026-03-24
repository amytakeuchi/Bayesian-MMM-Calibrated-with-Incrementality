"""
Saturation Transform (Hill Function)
====================================

Implements the Hill saturation curve to model diminishing returns.

Intuition:
- Early spend increases response strongly.
- After a point, additional spend yields smaller marginal gains.

Hill function:
    f(x) = x^alpha / (x^alpha + theta^alpha)
Parameters:
- alpha: curve steepness (shape)
- theta: half-saturation point (scale; spend level where response is ~0.5)

Why this file exists:
- Saturation is central to MMM decision-making (marginal ROI curves).
- Keeping it isolated makes it easy to swap in other forms (log, spline, etc.).

Design choice (canonical repo):
- Hill parameters are fixed by default to reduce identifiability issues.
- Calibration focuses on beta (effect size) rather than learning every nonlinearity at once.
"""

from __future__ import annotations
import numpy as np


def hill_saturation(X: np.ndarray, alpha: float, theta: float) -> np.ndarray:
    """
    Hill saturation applied elementwise to X.
    Output in [0, 1).
    """
    X = np.maximum(X, 0.0)
    num = np.power(X, alpha)
    den = num + np.power(theta, alpha) + 1e-12
    return num / den
