"""Configurable rolling window feature extraction for sensor time-series."""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd


def compute_rolling_trend_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute OLS linear regression slope over sliding time window: beta = Cov(t, x) / Var(t).

    Args:
        series: Input 1D time-series.
        window: Window size in number of steps.

    Returns:
        Sliding slope series.
    """
    if window < 2 or len(series) < window:
        return pd.Series(np.nan, index=series.index)

    # Standard independent time indices for window: t = [0, 1, ..., W-1]
    t = np.arange(window, dtype=np.float64)
    t_mean = (window - 1.0) / 2.0
    t_dev = t - t_mean
    denom = np.sum(t_dev ** 2)

    if denom <= 0:
        return pd.Series(np.nan, index=series.index)

    # Convolution with t_dev computes sum((t - t_bar) * x)
    # Reverse kernel because np.convolve flips the kernel
    kernel = t_dev[::-1] / denom

    vals = series.to_numpy(dtype=np.float64)
    # Valid convolution matches ending at window-1
    slopes = np.full(len(series), np.nan, dtype=np.float64)
    valid_conv = np.convolve(vals, kernel, mode="valid")
    slopes[window - 1:] = valid_conv

    return pd.Series(slopes, index=series.index)


class RollingFeatureExtractor:
    """Configurable multi-window rolling feature extractor.

    Supports configurable window sizes (e.g. 10, 30, 60 samples or seconds)
    without hardcoding any specific sampling frequency.
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        features: Optional[List[str]] = None,
        min_periods: Optional[int] = None,
    ):
        """
        Args:
            windows: List of window lengths in steps (e.g. [15, 60, 300]).
            features: List of statistics to extract:
                      ['mean', 'std', 'median', 'min', 'max', 'iqr', 'slope', 'variance', 'snr']
            min_periods: Minimum non-null samples required in window.
        """
        self.windows = windows or [15, 60]
        self.features = features or ["mean", "std", "min", "max", "slope", "variance"]
        self.min_periods = min_periods

    def extract_features(self, series: pd.Series, prefix: str = "") -> pd.DataFrame:
        """Extract rolling features for a given 1D series.

        Args:
            series: Sensor measurement series.
            prefix: Column name prefix for clarity (e.g. 'pressure', 'temp_0').

        Returns:
            DataFrame containing all rolling feature columns.
        """
        pfx = f"{prefix}_" if prefix else ""
        out_df = pd.DataFrame(index=series.index)

        for w in self.windows:
            mp = self.min_periods if self.min_periods is not None else max(2, w // 2)
            roller = series.rolling(window=w, min_periods=mp)

            if "mean" in self.features:
                out_df[f"{pfx}roll_mean_w{w}"] = roller.mean()

            if "std" in self.features:
                out_df[f"{pfx}roll_std_w{w}"] = roller.std()

            if "median" in self.features:
                out_df[f"{pfx}roll_median_w{w}"] = roller.median()

            if "min" in self.features:
                out_df[f"{pfx}roll_min_w{w}"] = roller.min()

            if "max" in self.features:
                out_df[f"{pfx}roll_max_w{w}"] = roller.max()

            if "variance" in self.features:
                out_df[f"{pfx}roll_var_w{w}"] = roller.var()

            if "iqr" in self.features:
                q75 = roller.quantile(0.75)
                q25 = roller.quantile(0.25)
                out_df[f"{pfx}roll_iqr_w{w}"] = q75 - q25

            if "slope" in self.features:
                out_df[f"{pfx}roll_slope_w{w}"] = compute_rolling_trend_slope(series, window=w)

            if "snr" in self.features:
                r_mean = roller.mean().abs()
                r_std = roller.std()
                safe_std = np.where(r_std > 1e-6, r_std, np.nan)
                out_df[f"{pfx}roll_snr_w{w}"] = r_mean / safe_std

        return out_df

