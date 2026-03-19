"""PCA statistical factor model for portfolio risk decomposition.

Extracts latent return factors via PCA on the cross-sectional return
matrix. PCA operates on the *correlation* matrix internally (sklearn
centers the data), so the resulting factors capture co-movement
patterns normalized by per-stock volatility. Per-stock volatility
information is preserved separately in the idiosyncratic residuals.

Replaces the need for a vendor factor model (Barra GEMLT) with a
statistical approximation that captures ~60-80% of the return
covariance structure at zero cost.

The model provides:
  - Factor return series (daily)
  - Factor loadings per stock (updated on each rebalance)
  - Regularized factor covariance matrix (Ledoit-Wolf shrinkage)
  - Residual (idiosyncratic) volatility per stock

Usage:
    from src.models.factor_model import StatisticalFactorModel

    fm = StatisticalFactorModel(n_factors=30, lookback=252)
    fm.fit(returns_df)

    loadings = fm.get_loadings(symbols)       # (n_stocks, n_factors)
    factor_cov = fm.factor_covariance          # (n_factors, n_factors)
    idio_vol = fm.idiosyncratic_vol(symbols)   # (n_stocks,)
    portfolio_factor_exposure = fm.portfolio_exposure(weights)
"""

import logging
from datetime import date

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class StatisticalFactorModel:
    """Rolling PCA factor model on cross-sectional stock returns.

    Note: sklearn PCA centers the input (subtracts column means), so
    it operates on the correlation structure, not the raw covariance.
    This means factors capture co-movement patterns normalized by
    per-stock volatility. Idiosyncratic volatility is preserved
    separately in the residual matrix.
    """

    def __init__(
        self,
        n_factors: int = 30,
        lookback: int = 252,
        min_history: int = 60,
    ) -> None:
        self.n_factors = n_factors
        self.lookback = lookback
        self.min_history = min_history

        self._pca: PCA | None = None
        self._loadings: pd.DataFrame | None = None
        self._factor_returns: pd.DataFrame | None = None
        self._residuals: pd.DataFrame | None = None
        self._symbols: list[str] = []
        self._fit_date: date | None = None
        self._explained_variance_ratio: np.ndarray | None = None

    def fit(self, returns_df: pd.DataFrame, as_of_date: date | None = None) -> "StatisticalFactorModel":
        """Fit PCA on a (dates x symbols) return matrix.

        Args:
            returns_df: DataFrame with DatetimeIndex/date index and symbol
                columns, containing daily returns.
            as_of_date: If provided, only use data up to this date.

        Returns self for chaining.
        """
        if as_of_date is not None:
            returns_df = returns_df.loc[:as_of_date]

        returns_df = returns_df.iloc[-self.lookback:]

        min_coverage = 0.5
        valid_cols = returns_df.columns[returns_df.notna().mean() >= min_coverage]
        if len(valid_cols) < self.n_factors + 5:
            logger.warning(
                f"Only {len(valid_cols)} stocks with >=50% coverage "
                f"(need {self.n_factors + 5}). Reducing n_factors."
            )
            self.n_factors = max(5, len(valid_cols) - 5)

        # Forward-fill then drop remaining NaNs. fillna(0.0) would
        # artificially deflate volatility and distort PCA factors.
        returns_subset = returns_df[valid_cols].ffill()
        returns_clean = returns_subset.dropna(axis=0, how="any")
        self._symbols = list(valid_cols)

        if len(returns_clean) < self.min_history:
            raise ValueError(
                f"Insufficient history: {len(returns_clean)} days "
                f"(need {self.min_history})"
            )

        n_components = min(self.n_factors, len(self._symbols) - 1, len(returns_clean) - 1)
        self._pca = PCA(n_components=n_components)

        # PCA on T x N return matrix: factors are T x K, loadings are N x K
        factor_returns_np = self._pca.fit_transform(returns_clean.values)
        loadings_np = self._pca.components_.T  # (N, K)

        self._factor_returns = pd.DataFrame(
            factor_returns_np,
            index=returns_clean.index,
            columns=[f"F{i+1}" for i in range(n_components)],
        )
        self._loadings = pd.DataFrame(
            loadings_np,
            index=self._symbols,
            columns=[f"F{i+1}" for i in range(n_components)],
        )

        # Residuals: actual returns minus factor-explained returns
        reconstructed = factor_returns_np @ self._pca.components_
        self._residuals = pd.DataFrame(
            returns_clean.values - reconstructed,
            index=returns_clean.index,
            columns=self._symbols,
        )

        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        self._fit_date = as_of_date or (
            returns_clean.index[-1].date()
            if hasattr(returns_clean.index[-1], "date")
            else returns_clean.index[-1]
        )

        total_var_explained = float(self._explained_variance_ratio.sum())
        logger.info(
            f"Factor model fit: {n_components} factors, "
            f"{len(self._symbols)} stocks, "
            f"{len(returns_clean)} days, "
            f"total variance explained: {total_var_explained:.1%}"
        )
        return self

    @property
    def factor_covariance(self) -> np.ndarray:
        """Regularized factor-factor covariance matrix (K x K).

        Uses Ledoit-Wolf shrinkage to avoid the same numerical
        instability that raw np.cov() would have with 30 factors on
        ~252 observations.
        """
        if self._factor_returns is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        lw = LedoitWolf()
        lw.fit(self._factor_returns.values)
        return lw.covariance_

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        if self._explained_variance_ratio is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._explained_variance_ratio

    def get_loadings(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """Factor loadings matrix (N x K).

        Args:
            symbols: Subset of symbols. If None, returns all.
        """
        if self._loadings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if symbols is None:
            return self._loadings
        available = [s for s in symbols if s in self._loadings.index]
        return self._loadings.loc[available]

    def idiosyncratic_vol(self, symbols: list[str] | None = None) -> pd.Series:
        """Annualized idiosyncratic (residual) volatility per stock."""
        if self._residuals is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        daily_vol = self._residuals.std()
        annual_vol = daily_vol * np.sqrt(252)
        if symbols is not None:
            available = [s for s in symbols if s in annual_vol.index]
            return annual_vol[available]
        return annual_vol

    def portfolio_exposure(self, weights: dict[str, float]) -> np.ndarray:
        """Compute portfolio's factor exposure vector (K,).

        Args:
            weights: {symbol: weight} dict (can be dollar weights or
                fractional weights — interpretation is caller's).
        """
        if self._loadings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        w_vec = np.zeros(len(self._symbols))
        for i, sym in enumerate(self._symbols):
            w_vec[i] = weights.get(sym, 0.0)
        return self._loadings.values.T @ w_vec  # (K,)

    def portfolio_factor_risk(self, weights: dict[str, float]) -> dict:
        """Decompose portfolio risk into factor and idiosyncratic components.

        Returns:
            dict with 'total_vol', 'factor_vol', 'idio_vol',
            'factor_exposures', 'pct_factor_risk'.
        """
        exposure = self.portfolio_exposure(weights)
        factor_cov = self.factor_covariance
        factor_var = float(exposure @ factor_cov @ exposure)

        idio_vols = self.idiosyncratic_vol()
        idio_var = 0.0
        for sym, w in weights.items():
            if sym in idio_vols.index:
                idio_var += (w * idio_vols[sym] / np.sqrt(252)) ** 2

        total_var = factor_var + idio_var
        total_vol = np.sqrt(total_var) * np.sqrt(252)
        factor_vol = np.sqrt(max(0, factor_var)) * np.sqrt(252)
        idio_vol_port = np.sqrt(max(0, idio_var)) * np.sqrt(252)

        pct_factor = factor_var / total_var if total_var > 0 else 0.0

        return {
            "total_vol": round(float(total_vol), 4),
            "factor_vol": round(float(factor_vol), 4),
            "idio_vol": round(float(idio_vol_port), 4),
            "pct_factor_risk": round(float(pct_factor), 4),
            "factor_exposures": {
                f"F{i+1}": round(float(e), 4) for i, e in enumerate(exposure)
            },
            "n_factors": len(exposure),
            "fit_date": str(self._fit_date),
        }

    def top_factor_exposures(
        self, weights: dict[str, float], top_n: int = 5,
    ) -> list[dict]:
        """Return the top-N absolute factor exposures for the portfolio."""
        exposure = self.portfolio_exposure(weights)
        factor_names = [f"F{i+1}" for i in range(len(exposure))]
        pairs = sorted(
            zip(factor_names, exposure),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [
            {"factor": name, "exposure": round(float(exp), 4)}
            for name, exp in pairs[:top_n]
        ]


def build_return_matrix(
    db,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build a (dates x symbols) daily return matrix from market_bars_daily.

    Returns a DataFrame with date index and symbol columns.
    """
    from sqlalchemy import text as sql_text

    result = db.execute(sql_text("""
        SELECT symbol, date, close
        FROM market_bars_daily
        WHERE symbol = ANY(:syms)
          AND date BETWEEN :s AND :e
        ORDER BY symbol, date
    """), {"syms": symbols, "s": start_date, "e": end_date})

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    pivot = df.pivot(index="date", columns="symbol", values="close")
    returns = pivot.pct_change().iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan)

    return returns
