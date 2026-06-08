from __future__ import annotations

from difflib import get_close_matches

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from optimizer import (
    PortfolioError,
    custom_month_range,
    fetch_average_tbill_rate,
    period_to_full_month_range,
    run_optimization,
)


st.set_page_config(
    page_title="美股 Sharpe 最大化投組",
    page_icon="📈",
    layout="wide",
)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def build_weights_table(weights: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Ticker": weights.index,
            "Weight": weights.values,
            "Weight (%)": (weights.values * 100).round(2),
        }
    )
    return table


COMMON_BENCHMARKS = [
    "SPY",
    "VOO",
    "IVV",
    "QQQ",
    "DIA",
    "IWM",
    "VTI",
    "VT",
    "ACWI",
    "VEA",
    "VWO",
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "0050",
    "0056",
    "00878",
    "2330",
    "2317",
    "2454",
]


def benchmark_suggestions(query: str) -> list[str]:
    clean = query.strip().upper()
    if not clean:
        return COMMON_BENCHMARKS[:8]
    prefix_matches = [ticker for ticker in COMMON_BENCHMARKS if ticker.startswith(clean)]
    fuzzy_matches = get_close_matches(clean, COMMON_BENCHMARKS, n=8, cutoff=0.2)
    suggestions = []
    for ticker in [*prefix_matches, *fuzzy_matches]:
        if ticker not in suggestions:
            suggestions.append(ticker)
    if clean and clean not in suggestions:
        suggestions.insert(0, clean)
    return suggestions[:8]


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def parse_uploaded_tickers(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    data = pd.read_csv(uploaded_file)
    if data.empty:
        raise PortfolioError("上傳的 CSV 沒有資料。")

    ticker_column = None
    for column in data.columns:
        if str(column).strip().lower() in {"ticker", "symbol", "代號", "標的"}:
            ticker_column = column
            break
    if ticker_column is None:
        ticker_column = data.columns[0]

    return data[ticker_column].dropna().astype(str).tolist()


def negative_red_style(value: float) -> str:
    if isinstance(value, (int, float)) and value < 0:
        return "color: red"
    return ""


def build_analysis_table(result) -> pd.DataFrame:
    benchmark_label = result.benchmark_ticker or "比較大盤"
    metric_rows = [
        ("初始投資資金", "initial_capital", format_money),
        ("終期投資資金", "ending_capital", format_money),
        ("CAGR", "cagr", format_percent),
        ("年化報酬率", "annual_return", format_percent),
        ("年化標準差", "volatility", format_percent),
        ("最大回撤 %", "max_drawdown", format_percent),
        ("SHARPE 指數", "sharpe", lambda value: f"{value:.2f}"),
    ]
    rows = []
    for name, key, formatter in metric_rows:
        benchmark_value = ""
        if result.benchmark_metrics:
            benchmark_value = formatter(result.benchmark_metrics[key])
        rows.append((name, formatter(result.metrics[key]), benchmark_value))
    return pd.DataFrame(rows, columns=["名稱", "投組", benchmark_label])


def build_monthly_returns_table(result) -> pd.DataFrame:
    table = result.monthly_returns.copy()
    table.insert(0, "投組", result.portfolio_returns)
    if result.benchmark_ticker and result.benchmark_returns is not None:
        table[result.benchmark_ticker] = result.benchmark_returns
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "月份"
    return table


@st.cache_data(ttl=60 * 60)
def get_default_rf_rate(start_date, end_date) -> tuple[float, str]:
    return fetch_average_tbill_rate(start_date, end_date)


st.title("美股 Sharpe 最大化投組")

with st.sidebar:
    st.header("參數")
    tickers = st.text_area(
        "Ticker 清單",
        value="AAPL, MSFT, NVDA",
        help="可用逗號、空白或換行分隔，例如 AAPL, MSFT, NVDA。",
    )
    uploaded_tickers = st.file_uploader(
        "上傳 ticker CSV",
        type=["csv"],
        help="可上傳含 ticker/symbol 欄位的 CSV；若沒有這些欄位，會讀取第一欄。",
    )
    benchmark_query = st.text_input(
        "比較大盤標的",
        value="SPY",
        help="例如 SPY、QQQ、VTI、2330、0050；台股數字代號會自動嘗試 .TW 與 .TWO。",
    )
    suggestions = benchmark_suggestions(benchmark_query)
    benchmark_choice = benchmark_query.strip().upper()
    if suggestions:
        benchmark_choice = st.selectbox(
            "模糊搜尋建議",
            options=suggestions,
            index=0 if benchmark_query.strip().upper() not in suggestions else suggestions.index(benchmark_query.strip().upper()),
        )
    period = st.selectbox(
        "回測期間",
        options=["近一年", "近三年", "近五年", "近十年", "自訂"],
        index=3,
    )
    custom_start = None
    custom_end = None
    custom_selection = None
    if period == "自訂":
        years = list(range(1990, pd.Timestamp.today().year + 1))
        months = list(range(1, 13))
        start_col, end_col = st.columns(2)
        with start_col:
            start_year = st.selectbox("起始年", years, index=max(0, len(years) - 11))
            start_month = st.selectbox("起始月", months, index=5)
        with end_col:
            end_year = st.selectbox("結束年", years, index=len(years) - 1)
            end_month = st.selectbox("結束月", months, index=5)
        custom_selection = (start_year, start_month, end_year, end_month)

    try:
        if period == "自訂" and custom_selection is not None:
            rf_start_date, rf_end_date = custom_month_range(*custom_selection)
        else:
            rf_start_date, rf_end_date = period_to_full_month_range(period)
        default_rf_rate, rf_source = get_default_rf_rate(rf_start_date, rf_end_date)
    except PortfolioError as exc:
        default_rf_rate, rf_source = 0.0368, f"資料來源：預設值 3.68%；{exc}"

    initial_capital = st.number_input(
        "初始投資資金",
        min_value=1.0,
        value=100000.0,
        step=10000.0,
    )
    rf_percent = st.number_input(
        "年化無風險利率 (%)",
        min_value=0.0,
        max_value=30.0,
        value=round(default_rf_rate * 100, 3),
        step=0.05,
    )
    st.caption(rf_source)
    run_button = st.button("開始最佳化", type="primary", use_container_width=True)

st.caption("價格資料由 yfinance 取得，使用 adjusted price 計算月報酬率。")

if run_button:
    try:
        ticker_inputs = []
        ticker_inputs.extend(parse_uploaded_tickers(uploaded_tickers))
        if tickers.strip():
            ticker_inputs.append(tickers)
        if period == "自訂" and custom_selection is not None:
            custom_start, custom_end = custom_month_range(*custom_selection)

        with st.spinner("下載價格並計算最佳投組..."):
            result = run_optimization(
                raw_tickers=ticker_inputs,
                period_label=period,
                rf_annual=rf_percent / 100,
                initial_capital=initial_capital,
                custom_start=custom_start,
                custom_end=custom_end,
                benchmark_ticker=benchmark_choice,
            )

        weights_table = build_weights_table(result.weights)
        non_zero_weights = weights_table[weights_table["Weight"] > 0.0001].copy()

        st.caption(f"實際分析期間：{result.start_date:%Y-%m-%d} 到 {result.end_date:%Y-%m-%d}")

        st.subheader("分析表")
        st.dataframe(
            build_analysis_table(result),
            hide_index=True,
            use_container_width=True,
        )

        left, right = st.columns([1, 1])
        with left:
            st.subheader("最佳持股比重")
            st.dataframe(
                non_zero_weights[["Ticker", "Weight (%)"]],
                hide_index=True,
                use_container_width=True,
            )
            csv_bytes = non_zero_weights[["Ticker", "Weight (%)"]].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載 optimized_portfolio.csv",
                data=csv_bytes,
                file_name="optimized_portfolio.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with right:
            st.subheader("權重分布")
            weight_fig = px.bar(
                non_zero_weights,
                x="Ticker",
                y="Weight (%)",
                text="Weight (%)",
                labels={"Weight (%)": "Weight (%)"},
            )
            weight_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            weight_fig.update_layout(yaxis_ticksuffix="%", showlegend=False)
            st.plotly_chart(weight_fig, use_container_width=True)

        st.subheader("投組淨值曲線")
        equity_fig = go.Figure()
        equity_fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode="lines",
                name="投組",
                line=dict(color="#1f77b4", width=2),
            )
        )
        if result.benchmark_ticker and result.benchmark_equity_curve is not None:
            equity_fig.add_trace(
                go.Scatter(
                    x=result.benchmark_equity_curve.index,
                    y=result.benchmark_equity_curve.values,
                    mode="lines",
                    name=result.benchmark_ticker,
                    line=dict(color="red", width=2),
                )
            )
        equity_fig.update_layout(yaxis_title="Portfolio Value", xaxis_title="", legend_title_text="")
        st.plotly_chart(equity_fig, use_container_width=True)

        with st.expander("月報酬率資料"):
            st.dataframe(
                build_monthly_returns_table(result).style.format("{:.2%}").map(negative_red_style),
                use_container_width=True,
            )

    except PortfolioError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover
        st.error(f"發生未預期錯誤：{exc}")
else:
    st.info("輸入 ticker 後按下「開始最佳化」。")
