from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


class PortfolioError(ValueError):
    """Raised when portfolio input data cannot produce a usable result."""


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    monthly_returns: pd.DataFrame
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]


def parse_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    if isinstance(raw_tickers, str):
        pieces = raw_tickers.replace("\n", ",").replace(" ", ",").split(",")
    else:
        pieces = list(raw_tickers)

    tickers: list[str] = []
    for ticker in pieces:
        clean = str(ticker).strip().upper()
        if clean and clean not in tickers:
            tickers.append(clean)

    if len(tickers) < 2:
        raise PortfolioError("請至少輸入 2 個有效 ticker，才能進行投組最佳化。")
    return tickers


def period_to_start_date(period_label: str, today: date | None = None) -> date:
    today = today or date.today()
    years_by_label = {
        "近一年": 1,
        "近三年": 3,
        "近五年": 5,
        "近十年": 10,
    }
    years = years_by_label.get(period_label)
    if years is None:
        raise PortfolioError(f"不支援的回測期間：{period_label}")
    return date(today.year - years, today.month, today.day)


def download_adjusted_prices(
    tickers: list[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PortfolioError("缺少 yfinance 套件，請先執行 pip install -r requirements.txt。") from exc

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise PortfolioError("yfinance 未回傳價格資料，請確認 ticker 或日期區間。")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise PortfolioError("下載資料缺少 adjusted close 價格欄位。")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise PortfolioError("下載資料缺少 adjusted close 價格欄位。")
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    prices = prices.dropna(axis=1, how="all")

    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise PortfolioError(f"以下 ticker 無可用價格資料：{', '.join(missing)}")

    return prices[tickers]


def prices_to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise PortfolioError("價格資料為空。")

    clean_prices = prices.replace([np.inf, -np.inf, 0], np.nan).dropna(axis=1, how="any")
    removed = sorted(set(prices.columns) - set(clean_prices.columns))
    if removed:
        raise PortfolioError(f"以下 ticker 有缺值或無效價格，已無法納入最佳化：{', '.join(removed)}")

    monthly_prices = clean_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="any")

    if len(monthly_returns) < 12:
        raise PortfolioError("月報酬率資料少於 12 期，請拉長回測期間或更換 ticker。")
    if monthly_returns.shape[1] < 2:
        raise PortfolioError("可用 ticker 少於 2 個，無法進行投組最佳化。")

    return monthly_returns


def optimize_max_sharpe(monthly_returns: pd.DataFrame, rf_annual: float) -> pd.Series:
    if monthly_returns.empty:
        raise PortfolioError("月報酬率資料為空。")

    zero_volatility = monthly_returns.std(ddof=1)
    zero_volatility = zero_volatility[zero_volatility <= 0]
    if not zero_volatility.empty:
        raise PortfolioError(f"以下 ticker 月報酬率波動度為 0，無法計算 Sharpe：{', '.join(zero_volatility.index)}")

    mean_returns = monthly_returns.mean().to_numpy()
    covariance = monthly_returns.cov().to_numpy()
    diagonal_max = float(np.nanmax(np.diag(covariance)))
    ridge = max(diagonal_max, 1e-12) * 1e-10
    covariance = covariance + np.eye(monthly_returns.shape[1]) * ridge
    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    asset_count = monthly_returns.shape[1]

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_mean = float(np.dot(weights, mean_returns))
        portfolio_vol = float(np.sqrt(weights.T @ covariance @ weights))
        if portfolio_vol <= 0 or not np.isfinite(portfolio_vol):
            return 1e6
        return -((portfolio_mean - rf_monthly) / portfolio_vol) * np.sqrt(MONTHS_PER_YEAR)

    initial_weights = np.repeat(1 / asset_count, asset_count)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise PortfolioError(f"最佳化失敗：{result.message}")

    weights = pd.Series(result.x, index=monthly_returns.columns, name="Weight")
    weights = weights.clip(lower=0)
    weights = weights / weights.sum()
    return weights.sort_values(ascending=False)


def calculate_portfolio_performance(
    monthly_returns: pd.DataFrame,
    weights: pd.Series,
    rf_annual: float,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    aligned_returns = monthly_returns[weights.index]
    portfolio_returns = aligned_returns.mul(weights, axis=1).sum(axis=1)
    equity_curve = (1 + portfolio_returns).cumprod()

    years = len(portfolio_returns) / MONTHS_PER_YEAR
    total_return = float(equity_curve.iloc[-1] - 1)
    cagr = float(equity_curve.iloc[-1] ** (1 / years) - 1)
    volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1

    if volatility <= 0 or not np.isfinite(volatility):
        sharpe = np.nan
    else:
        sharpe = float((portfolio_returns.mean() - rf_monthly) / portfolio_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    metrics = {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "total_return": total_return,
    }
    return portfolio_returns, equity_curve, metrics


def run_optimization(
    raw_tickers: str | Iterable[str],
    period_label: str,
    rf_annual: float,
    end_date: date | None = None,
) -> OptimizationResult:
    tickers = parse_tickers(raw_tickers)
    end = end_date or date.today()
    start = period_to_start_date(period_label, today=end)
    prices = download_adjusted_prices(tickers, start_date=start, end_date=end)
    monthly_returns = prices_to_monthly_returns(prices)
    weights = optimize_max_sharpe(monthly_returns, rf_annual)
    portfolio_returns, equity_curve, metrics = calculate_portfolio_performance(
        monthly_returns=monthly_returns,
        weights=weights,
        rf_annual=rf_annual,
    )
    return OptimizationResult(
        weights=weights,
        monthly_returns=monthly_returns,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        metrics=metrics,
    )
