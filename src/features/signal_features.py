"""Signal processing and instantaneous feature extraction for sensor time-series."""

from typing import Optional
import numpy as np
import pandas as pd


def compute_rate_of_change(series: pd.Series, time_series: Optional[pd.Series] = None) -> pd.Series:
    """Compute first-order numerical derivative dx/dt with respect to time.

    Args:
        series: Measurement values.
        time_series: Timestamp series (seconds or datetime). If None, step delta dt=1 is assumed.

    Returns:
        Rate of change series.
    """
    diff_val = series.diff()
    if time_series is not None:
        if pd.api.types.is_datetime64_any_dtype(time_series):
            dt = time_series.diff().dt.total_seconds()
        else:
            dt = time_series.diff()
        # Prevent division by zero
        dt_safe = np.where(dt > 0, dt, np.nan)
        return diff_val / dt_safe
    return diff_val


def compute_acceleration(series: pd.Series, time_series: Optional[pd.Series] = None) -> pd.Series:
    """Compute second-order numerical derivative d2x/dt2."""
    roc = compute_rate_of_change(series, time_series)
    return compute_rate_of_change(roc, time_series)


def compute_instantaneous_snr(signal: pd.Series, noise_std: float) -> pd.Series:
    """Compute signal-to-noise ratio in decibels: 20 * log10(|signal| / noise_std)."""
    if noise_std <= 0:
        return pd.Series(np.nan, index=signal.index)
    snr_linear = signal.abs() / noise_std
    return 20.0 * np.log10(np.clip(snr_linear, a_min=1e-6, a_max=None))

