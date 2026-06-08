from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from calendar import monthrange
from typing import Iterable
from urllib.parse import quote

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
    benchmark_ticker: str | None
    benchmark_returns: pd.Series | None
    benchmark_equity_curve: pd.Series | None
    benchmark_metrics: dict[str, float] | None
    ignored_tickers: list[str]
    price_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    optimizer_engine: str
    optimizer_note: str
    start_date: date
    end_date: date


def parse_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    separators = ["\n", " ", "，", ";", "；"]
    if isinstance(raw_tickers, str):
        pieces = [raw_tickers]
    else:
        pieces = list(raw_tickers)

    tickers: list[str] = []
    for item in pieces:
        normalized = str(item)
        for separator in separators:
            normalized = normalized.replace(separator, ",")
        for ticker in normalized.split(","):
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


def month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


def shift_month(year: int, month: int, month_delta: int) -> tuple[int, int]:
    month_index = year * 12 + month - 1 + month_delta
    return month_index // 12, month_index % 12 + 1


def previous_month_start(value: date) -> date:
    year, month = shift_month(value.year, value.month, -1)
    return month_start(year, month)


def period_to_full_month_range(period_label: str, today: date | None = None) -> tuple[date, date]:
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

    end_year, end_month = shift_month(today.year, today.month, -1)
    start_year, start_month = shift_month(end_year, end_month, -(years * 12 - 1))
    return month_start(start_year, start_month), month_end(end_year, end_month)


def parse_optional_ticker(raw_ticker: str | None) -> str | None:
    if raw_ticker is None or not raw_ticker.strip():
        return None
    separators = ["\n", " ", "，", ";", "；"]
    normalized = raw_ticker
    for separator in separators:
        normalized = normalized.replace(separator, ",")
    for ticker in normalized.split(","):
        clean = ticker.strip().upper()
        if clean:
            return clean
    return None


def ticker_download_candidates(ticker: str) -> list[str]:
    clean = ticker.strip().upper()
    if clean.isdigit() and 4 <= len(clean) <= 6:
        return [f"{clean}.TW", f"{clean}.TWO"]
    return [clean]


def custom_month_range(start_year: int, start_month: int, end_year: int, end_month: int) -> tuple[date, date]:
    start = month_start(start_year, start_month)
    end = month_end(end_year, end_month)
    if start > end:
        raise PortfolioError("自訂起始月份不可晚於結束月份。")
    return start, end


def fetch_latest_tbill_rate(default_rate: float = 0.0368) -> tuple[float, str]:
    try:
        prices = fetch_yahoo_chart_prices("^IRX", date.today() - timedelta(days=30), date.today(), interval="1d")
        latest = prices["^IRX"].dropna()
        if latest.empty:
            raise PortfolioError("無可用 ^IRX 資料。")

        rate = float(latest.iloc[-1]) / 100
        as_of = latest.index[-1].strftime("%Y-%m-%d")
        return rate, f"^IRX 13 WEEK TREASURY BILL as of {as_of}"
    except Exception:
        return default_rate, "預設值 3.68%；無法自動取得 ^IRX"


def fetch_average_tbill_rate(
    start_date: date,
    end_date: date,
    default_rate: float = 0.0368,
) -> tuple[float, str]:
    try:
        prices = fetch_yahoo_chart_prices("^IRX", start_date=start_date, end_date=end_date, interval="1d")
        monthly_rates = prices["^IRX"].resample("ME").last().dropna() / 100
        start_boundary = pd.Timestamp(month_end(start_date.year, start_date.month))
        end_boundary = pd.Timestamp(month_end(end_date.year, end_date.month))
        monthly_rates = monthly_rates.loc[(monthly_rates.index >= start_boundary) & (monthly_rates.index <= end_boundary)]
        if monthly_rates.empty:
            raise PortfolioError("無可用 ^IRX 月資料。")

        rate = float(monthly_rates.mean())
        first_month = monthly_rates.index[0].strftime("%Y-%m")
        last_month = monthly_rates.index[-1].strftime("%Y-%m")
        return rate, f"資料來源：Yahoo Finance ^IRX 13-week T-bill，期間平均 {first_month} ~ {last_month}"
    except Exception:
        return default_rate, "資料來源：預設值 3.68%；無法自動取得回測期間平均 ^IRX"


def fetch_yahoo_chart_prices(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> pd.DataFrame:
    import requests

    period1 = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp())
    encoded_ticker = quote(ticker, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}"
    response = requests.get(
        url,
        params={
            "period1": period1,
            "period2": period2,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise PortfolioError(f"Yahoo Finance chart API 未回傳 {ticker} 資料。")

    item = result[0]
    timestamps = item.get("timestamp", [])
    closes = item.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    if not timestamps or not closes:
        raise PortfolioError(f"Yahoo Finance chart API 缺少 {ticker} close 資料。")

    index = pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert(None)
    prices = pd.DataFrame({ticker: pd.to_numeric(pd.Series(closes), errors="coerce").to_numpy()}, index=index)
    return prices.dropna(how="all")


def download_adjusted_prices(
    tickers: list[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PortfolioError("缺少 yfinance 套件，請先執行 pip install -r requirements.txt。") from exc

    download_end = end_date
    if isinstance(end_date, date):
        download_end = end_date + timedelta(days=1)

    primary_download_tickers = [ticker_download_candidates(ticker)[0] for ticker in tickers]
    raw = yf.download(
        tickers=primary_download_tickers,
        start=start_date,
        end=download_end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise PortfolioError("yfinance 未回傳價格資料，請確認 ticker 或日期區間。")

    downloaded_prices = _extract_close_prices(raw, primary_download_tickers)
    prices = pd.DataFrame(index=downloaded_prices.index)
    for ticker, download_ticker in zip(tickers, primary_download_tickers):
        if download_ticker in downloaded_prices:
            prices[ticker] = downloaded_prices[download_ticker]

    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        fallback_prices = []
        for ticker in missing:
            for download_ticker in ticker_download_candidates(ticker):
                single_raw = yf.download(
                    tickers=download_ticker,
                    start=start_date,
                    end=download_end,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=False,
                )
                if single_raw.empty:
                    continue
                try:
                    single_prices = _extract_close_prices(single_raw, [download_ticker])
                except PortfolioError:
                    continue
                if download_ticker in single_prices:
                    fallback_prices.append(single_prices[[download_ticker]].rename(columns={download_ticker: ticker}))
                    break

        if fallback_prices:
            prices = pd.concat([prices, *fallback_prices], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()]

    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        prices = prices.drop(columns=[ticker for ticker in missing if ticker in prices.columns], errors="ignore")
        if prices.empty:
            raise PortfolioError(f"所有 ticker 都沒有可用價格資料：{', '.join(tickers)}")

    valid_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    return prices[valid_tickers].sort_index()


def _extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
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
    prices.columns = [str(column).upper() for column in prices.columns]

    rename_map = {}
    available_by_upper = {str(column).upper(): column for column in prices.columns}
    for ticker in tickers:
        if ticker.upper() in available_by_upper:
            rename_map[available_by_upper[ticker.upper()]] = ticker
    return prices.rename(columns=rename_map)


def prices_to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise PortfolioError("價格資料為空。")

    clean_prices = prices.replace([np.inf, -np.inf, 0], np.nan).dropna(axis=1, how="all")
    removed = sorted(set(prices.columns) - set(clean_prices.columns))
    if removed:
        raise PortfolioError(f"以下 ticker 沒有有效價格，無法納入最佳化：{', '.join(removed)}")

    monthly_prices = clean_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change()

    if len(monthly_returns) < 12:
        raise PortfolioError("月報酬率資料少於 12 期，請拉長回測期間或更換 ticker。")
    if monthly_returns.shape[1] < 2:
        raise PortfolioError("可用 ticker 少於 2 個，無法進行投組最佳化。")

    return monthly_returns


def calculate_price_ranges(prices: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = {}
    for ticker in prices.columns:
        valid_prices = prices[ticker].replace([np.inf, -np.inf, 0], np.nan).dropna()
        if not valid_prices.empty:
            ranges[ticker] = (valid_prices.index.min(), valid_prices.index.max())
    return ranges


def filter_monthly_returns(
    monthly_returns: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    start_boundary = pd.Timestamp(month_end(start_date.year, start_date.month))
    end_boundary = pd.Timestamp(month_end(end_date.year, end_date.month))
    filtered = monthly_returns.loc[(monthly_returns.index >= start_boundary) & (monthly_returns.index <= end_boundary)]
    if len(filtered) < 12:
        raise PortfolioError("指定月份範圍內月報酬率資料少於 12 期，請拉長回測期間或更換 ticker。")
    return filtered


def optimize_max_sharpe(
    monthly_returns: pd.DataFrame,
    rf_annual: float,
    optimizer_engine: str = "ffn-compatible",
) -> tuple[pd.Series, str, str]:
    if monthly_returns.empty:
        raise PortfolioError("月報酬率資料為空。")

    monthly_returns = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if monthly_returns.empty:
        raise PortfolioError("月報酬率資料清理後為空。")

    zero_volatility = monthly_returns.std(ddof=1)
    zero_volatility = zero_volatility[zero_volatility <= 0]
    if not zero_volatility.empty:
        raise PortfolioError(f"以下 ticker 月報酬率波動度為 0，無法計算 Sharpe：{', '.join(zero_volatility.index)}")

    if optimizer_engine == "ffn-compatible":
        try:
            weights = optimize_with_ffn(monthly_returns)
            return weights, "ffn-compatible", "使用 ffn.core.calc_mean_var_weights；權重最佳化 rf=0，對齊原始 Python。"
        except Exception as exc:
            fallback_weights = optimize_stable_max_sharpe(monthly_returns, rf_annual)
            return fallback_weights, "stable-fallback", f"ffn 最佳化失敗，已改用穩定 fallback；ffn 錯誤：{exc}"

    weights = optimize_stable_max_sharpe(monthly_returns, rf_annual)
    return weights, "stable", "使用本工具穩定最大 Sharpe 模式。"


def optimize_with_ffn(monthly_returns: pd.DataFrame) -> pd.Series:
    import ffn

    weights = ffn.core.calc_mean_var_weights(
        monthly_returns,
        weight_bounds=(0.0, 1.0),
        rf=0.0,
        covar_method="ledoit-wolf",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    weights = weights.clip(lower=0)
    if weights.sum() <= 0:
        raise PortfolioError("ffn 回傳的權重總和小於等於 0。")
    weights = weights / weights.sum()
    return weights.sort_values(ascending=False)


def optimize_stable_max_sharpe(monthly_returns: pd.DataFrame, rf_annual: float) -> pd.Series:
    mean_returns = monthly_returns.mean().to_numpy()
    covariance = monthly_returns.cov().to_numpy()
    diagonal_max = float(np.nanmax(np.diag(covariance)))
    ridge = max(diagonal_max, 1e-12) * 1e-8
    covariance = covariance + np.eye(monthly_returns.shape[1]) * ridge
    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    asset_count = monthly_returns.shape[1]
    observation_count = monthly_returns.shape[0]

    if asset_count > 150 or asset_count >= observation_count:
        return optimize_large_universe_max_sharpe(monthly_returns, mean_returns, covariance, rf_monthly)

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_mean = float(np.dot(weights, mean_returns))
        portfolio_vol = float(np.sqrt(weights.T @ covariance @ weights))
        if portfolio_vol <= 0 or not np.isfinite(portfolio_vol):
            return 1e6
        return -((portfolio_mean - rf_monthly) / portfolio_vol) * np.sqrt(MONTHS_PER_YEAR)

    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    initial_weights = build_initial_weight_guesses(mean_returns, covariance, asset_count)

    best_result = None
    best_score = np.inf
    failure_messages = []
    for guess in initial_weights:
        result = minimize(
            negative_sharpe,
            guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        score = negative_sharpe(result.x)
        feasible = (
            np.isfinite(score)
            and np.all(result.x >= -1e-6)
            and np.all(result.x <= 1 + 1e-6)
            and abs(float(np.sum(result.x)) - 1.0) <= 1e-5
        )
        if result.success or feasible:
            if score < best_score:
                best_score = score
                best_result = result
        else:
            failure_messages.append(str(result.message))

    if best_result is None:
        details = sorted(set(failure_messages))
        detail_text = f"；原因：{' / '.join(details)}" if details else ""
        raise PortfolioError(f"最佳化失敗：請縮短 ticker 清單、拉長回測期間，或移除高度相關/資料不穩定標的{detail_text}")

    weights = pd.Series(best_result.x, index=monthly_returns.columns, name="Weight")
    weights = weights.clip(lower=0)
    weights = weights / weights.sum()
    return weights.sort_values(ascending=False)


def build_initial_weight_guesses(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    asset_count: int,
) -> list[np.ndarray]:
    guesses = [np.repeat(1 / asset_count, asset_count)]

    asset_sharpe_order = np.argsort(-(mean_returns / np.sqrt(np.clip(np.diag(covariance), 1e-12, None))))
    unit_guess_indexes = asset_sharpe_order[: min(asset_count, 25)]
    for index in unit_guess_indexes:
        unit = np.zeros(asset_count)
        unit[index] = 1.0
        guesses.append(unit)

    positive_returns = np.clip(mean_returns, 0, None)
    if positive_returns.sum() > 0:
        guesses.append(positive_returns / positive_returns.sum())

    volatilities = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    inverse_vol = 1 / volatilities
    guesses.append(inverse_vol / inverse_vol.sum())

    unique_guesses = []
    for guess in guesses:
        clean_guess = np.clip(np.asarray(guess, dtype=float), 0, 1)
        total = clean_guess.sum()
        if total <= 0:
            continue
        clean_guess = clean_guess / total
        if not any(np.allclose(clean_guess, existing, atol=1e-10) for existing in unique_guesses):
            unique_guesses.append(clean_guess)
    return unique_guesses


def optimize_large_universe_max_sharpe(
    monthly_returns: pd.DataFrame,
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    rf_monthly: float,
) -> pd.Series:
    excess_returns = mean_returns - rf_monthly
    asset_count = monthly_returns.shape[1]

    shrinkage_covariance = shrink_covariance(covariance, shrinkage=0.35)
    try:
        raw_weights = np.linalg.pinv(shrinkage_covariance, rcond=1e-8) @ excess_returns
    except np.linalg.LinAlgError:
        raw_weights = excess_returns / np.sqrt(np.clip(np.diag(shrinkage_covariance), 1e-12, None))

    weights = np.clip(raw_weights, 0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        volatilities = np.sqrt(np.clip(np.diag(shrinkage_covariance), 1e-12, None))
        single_asset_scores = excess_returns / volatilities
        best_index = int(np.nanargmax(single_asset_scores))
        weights = np.zeros(asset_count)
        weights[best_index] = 1.0
    else:
        weights = weights / weights.sum()

    weights = remove_tiny_weights(weights)
    return pd.Series(weights, index=monthly_returns.columns, name="Weight").sort_values(ascending=False)


def shrink_covariance(covariance: np.ndarray, shrinkage: float) -> np.ndarray:
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1 - shrinkage) * covariance + shrinkage * diagonal
    diagonal_max = float(np.nanmax(np.diag(shrunk)))
    ridge = max(diagonal_max, 1e-12) * 1e-6
    return shrunk + np.eye(shrunk.shape[0]) * ridge


def remove_tiny_weights(weights: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    clean_weights = np.where(weights >= threshold, weights, 0.0)
    if clean_weights.sum() <= 0:
        return weights / weights.sum()
    return clean_weights / clean_weights.sum()


def calculate_portfolio_performance(
    monthly_returns: pd.DataFrame,
    weights: pd.Series,
    rf_annual: float,
    initial_capital: float,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    aligned_returns = monthly_returns[weights.index]
    portfolio_returns = aligned_returns.mul(weights, axis=1).sum(axis=1)
    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()

    years = len(portfolio_returns) / MONTHS_PER_YEAR
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)
    cagr = float((equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1)
    annual_return = float(portfolio_returns.mean() * MONTHS_PER_YEAR)
    volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min())

    if volatility <= 0 or not np.isfinite(volatility):
        sharpe = np.nan
    else:
        sharpe = float((portfolio_returns.mean() - rf_monthly) / portfolio_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": float(equity_curve.iloc[-1]),
        "cagr": cagr,
        "annual_return": annual_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "total_return": total_return,
    }
    return portfolio_returns, equity_curve, metrics


def calculate_single_asset_performance(
    monthly_returns: pd.Series,
    rf_annual: float,
    initial_capital: float,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    clean_returns = monthly_returns.dropna()
    if len(clean_returns) < 12:
        raise PortfolioError("比較大盤標的的月報酬率資料少於 12 期。")

    equity_curve = initial_capital * (1 + clean_returns).cumprod()
    years = len(clean_returns) / MONTHS_PER_YEAR
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)
    cagr = float((equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1)
    annual_return = float(clean_returns.mean() * MONTHS_PER_YEAR)
    volatility = float(clean_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min())

    if volatility <= 0 or not np.isfinite(volatility):
        sharpe = np.nan
    else:
        sharpe = float((clean_returns.mean() - rf_monthly) / clean_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": float(equity_curve.iloc[-1]),
        "cagr": cagr,
        "annual_return": annual_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "total_return": total_return,
    }
    return clean_returns, equity_curve, metrics


def run_optimization(
    raw_tickers: str | Iterable[str],
    period_label: str,
    rf_annual: float,
    initial_capital: float = 100000,
    end_date: date | None = None,
    custom_start: date | None = None,
    custom_end: date | None = None,
    benchmark_ticker: str | None = "SPY",
    optimizer_engine: str = "ffn-compatible",
) -> OptimizationResult:
    tickers = parse_tickers(raw_tickers)
    benchmark = parse_optional_ticker(benchmark_ticker)
    if initial_capital <= 0:
        raise PortfolioError("初始投資資金必須大於 0。")

    if period_label == "自訂":
        if custom_start is None or custom_end is None:
            raise PortfolioError("請提供自訂起訖月份。")
        start, end = custom_start, custom_end
    else:
        end_reference = end_date or date.today()
        start, end = period_to_full_month_range(period_label, today=end_reference)

    download_start = previous_month_start(start)
    download_tickers = list(tickers)
    if benchmark and benchmark not in download_tickers:
        download_tickers.append(benchmark)

    prices = download_adjusted_prices(download_tickers, start_date=download_start, end_date=end)
    ignored_tickers = [ticker for ticker in download_tickers if ticker not in prices.columns]
    price_ranges = calculate_price_ranges(prices)
    valid_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    if len(valid_tickers) < 2:
        ignored_text = f"；已忽略：{', '.join(ignored_tickers)}" if ignored_tickers else ""
        raise PortfolioError(f"可用投組 ticker 少於 2 個，無法進行投組最佳化{ignored_text}。")

    all_monthly_returns = filter_monthly_returns(prices_to_monthly_returns(prices), start, end)
    incomplete_return_tickers = sorted(all_monthly_returns.columns[all_monthly_returns.isna().any()].tolist())
    if incomplete_return_tickers:
        all_monthly_returns = all_monthly_returns.drop(columns=incomplete_return_tickers)
        ignored_tickers.extend([ticker for ticker in incomplete_return_tickers if ticker not in ignored_tickers])

    valid_tickers = [ticker for ticker in valid_tickers if ticker in all_monthly_returns.columns]
    if len(valid_tickers) < 2:
        ignored_text = f"；已忽略：{', '.join(ignored_tickers)}" if ignored_tickers else ""
        raise PortfolioError(f"可用投組 ticker 少於 2 個，無法進行投組最佳化{ignored_text}。")

    monthly_returns = all_monthly_returns[valid_tickers]
    weights, resolved_engine, optimizer_note = optimize_max_sharpe(
        monthly_returns,
        rf_annual,
        optimizer_engine=optimizer_engine,
    )
    portfolio_returns, equity_curve, metrics = calculate_portfolio_performance(
        monthly_returns=monthly_returns,
        weights=weights,
        rf_annual=rf_annual,
        initial_capital=initial_capital,
    )
    benchmark_returns = None
    benchmark_equity_curve = None
    benchmark_metrics = None
    if benchmark and benchmark in all_monthly_returns:
        benchmark_returns, benchmark_equity_curve, benchmark_metrics = calculate_single_asset_performance(
            all_monthly_returns[benchmark],
            rf_annual=rf_annual,
            initial_capital=initial_capital,
        )
    elif benchmark and benchmark not in ignored_tickers:
        ignored_tickers.append(benchmark)

    return OptimizationResult(
        weights=weights,
        monthly_returns=monthly_returns,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        benchmark_ticker=benchmark,
        benchmark_returns=benchmark_returns,
        benchmark_equity_curve=benchmark_equity_curve,
        benchmark_metrics=benchmark_metrics,
        ignored_tickers=ignored_tickers,
        price_ranges=price_ranges,
        optimizer_engine=resolved_engine,
        optimizer_note=optimizer_note,
        start_date=start,
        end_date=end,
    )
