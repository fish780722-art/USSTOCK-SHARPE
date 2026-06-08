from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from optimizer import PortfolioError, run_optimization


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


st.title("美股 Sharpe 最大化投組")

with st.sidebar:
    st.header("參數")
    tickers = st.text_area(
        "Ticker 清單",
        value="AAPL, MSFT, NVDA",
        help="可用逗號、空白或換行分隔，例如 AAPL, MSFT, NVDA。",
    )
    period = st.selectbox(
        "回測期間",
        options=["近一年", "近三年", "近五年", "近十年"],
        index=3,
    )
    rf_percent = st.number_input(
        "年化無風險利率 (%)",
        min_value=0.0,
        max_value=30.0,
        value=3.68,
        step=0.05,
    )
    run_button = st.button("開始最佳化", type="primary", use_container_width=True)

st.caption("價格資料由 yfinance 取得，使用 adjusted price 計算月報酬率。")

if run_button:
    try:
        with st.spinner("下載價格並計算最佳投組..."):
            result = run_optimization(
                raw_tickers=tickers,
                period_label=period,
                rf_annual=rf_percent / 100,
            )

        weights_table = build_weights_table(result.weights)
        non_zero_weights = weights_table[weights_table["Weight"] > 0.0001].copy()

        metric_cols = st.columns(4)
        metric_cols[0].metric("CAGR", format_percent(result.metrics["cagr"]))
        metric_cols[1].metric("年化波動度", format_percent(result.metrics["volatility"]))
        metric_cols[2].metric("Sharpe Ratio", f"{result.metrics['sharpe']:.2f}")
        metric_cols[3].metric("累積報酬", format_percent(result.metrics["total_return"]))

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
        equity_df = result.equity_curve.rename("Portfolio Value").reset_index()
        equity_df.columns = ["Date", "Portfolio Value"]
        equity_fig = px.line(equity_df, x="Date", y="Portfolio Value")
        equity_fig.update_layout(yaxis_title="Growth of $1", xaxis_title="")
        st.plotly_chart(equity_fig, use_container_width=True)

        with st.expander("月報酬率資料"):
            st.dataframe(
                result.monthly_returns.style.format("{:.2%}"),
                use_container_width=True,
            )

    except PortfolioError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover
        st.error(f"發生未預期錯誤：{exc}")
else:
    st.info("輸入 ticker 後按下「開始最佳化」。")

