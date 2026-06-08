# 美股 Sharpe 最大化投組 Streamlit App

這是一個以 Streamlit 建立的美股投組最佳化工具。使用者只需要輸入 ticker，系統會透過 yfinance 取得免費 adjusted price，轉成月報酬率後計算最大化 Sharpe ratio 的 long-only 投組權重。

## 功能

- 輸入自訂美股 ticker 清單
- 回測期間可選近一年、近三年、近五年、近十年，預設近十年
- 年化無風險利率可自行調整，預設 3.68%
- 使用 adjusted price 計算月報酬率
- 輸出 CAGR、年化波動度、Sharpe ratio、累積報酬
- 顯示最佳持股比重、權重圖表、投組淨值曲線
- 下載 `optimized_portfolio.csv`

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
streamlit run app.py
```

## 使用方式

1. 在側邊欄輸入 ticker，例如：

```text
AAPL, MSFT, NVDA
```

2. 選擇回測期間。
3. 設定年化無風險利率。
4. 按下「開始最佳化」。

## 計算邏輯

- 資料來源：yfinance
- 價格口徑：adjusted price
- 頻率：日價格轉月末價格，再計算月報酬率
- 限制：long-only，單一標的權重介於 0% 到 100%，總權重等於 100%
- 目標：最大化年化 Sharpe ratio

## 注意事項

- yfinance 是免費資料來源，可能因網路、Yahoo Finance 限流或 ticker 資料狀態而失敗。
- 第一版不支援本機 CSV 上傳、不支援放空、不支援槓桿、不支援單股權重上限。
- 最佳化結果是歷史資料回測，不代表未來績效。

