"""
Adstock Transform
=================

Implements geometric adstock for marketing spend time series.

Adstock intuition:
- Advertising impact persists beyond the spend week/day.
- Geometric adstock approximates carryover with an exponential decay.

Mathematically:
- adstock[t] = spend[t] + lambda * adstock[t-1]
- lambda in [0, 1). Higher lambda => longer memory.

Why this file exists:
- Keeps transformations reusable and testable
- Makes MMM structure easy to inspect and explain in interviews

Output:
- A transformed spend matrix with the same shape as input (T, C)
"""

from __future__ import annotations
import numpy as np


def geometric_adstock_2d(X: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    Geometric adstock for 2D matrix: X shape (T, C), lambdas shape (C,).
    """
    # Get the number of time periods (T) and the number of channels (C)
    T, C = X.shape
    # Initialize an empty matrix of zeros to store the results
    out = np.zeros_like(X, dtype=float)
    # Initialize a 'carry' vector to track the decaying impact from previous steps
    carry = np.zeros(C, dtype=float)
    # Loop through each time period (e.g., each day or week)
    for t in range(T):
        # Calculate current impact: New spend + (decay rate * leftover impact)
        # Formula: $adstock_{t} = spend_{t} + \lambda \cdot adstock_{t-1}$
        carry = X[t] + lambdas * carry
        # Store the calculated adstock value for this time period in the output matrix
        out[t] = carry
        # Return the full matrix of transformed spend values
    return out
