# 美股 Sharpe 最大化投組 Streamlit App

這是一個以 Streamlit 建立的美股投組最佳化工具。使用者只需要輸入 ticker，系統會透過 yfinance 取得免費 adjusted price，轉成月報酬率後計算最大化 Sharpe ratio 的 long-only 投組權重。

## 功能

- 輸入自訂美股 ticker 清單
- 可上傳 ticker CSV，預設讀取 `ticker` / `symbol` 欄位，若沒有則讀第一欄
- 回測期間可選近一年、近三年、近五年、近十年、自訂月份，預設近十年
- 年化無風險利率預設抓取 Yahoo Finance `^IRX` 13 WEEK TREASURY BILL，仍可自行調整
- 可自訂初始投資資金
- 使用 adjusted price 計算月報酬率
- 輸出初始投資資金、終期投資資金、CAGR、投組年化報酬率、投組年化標準差、最大回撤、Sharpe ratio
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
3. 若選擇自訂，設定起訖年月；例如 2015 年 6 月到 2025 年 6 月，會使用 2015-06-01 到 2025-06-30。
4. 設定初始投資資金與年化無風險利率。
5. 按下「開始最佳化」。

## 計算邏輯

- 資料來源：yfinance
- 價格口徑：adjusted price
- 頻率：日價格轉月末價格，再計算月報酬率
- 預設期間：以最近月份為結束月份，例如 2026 年 6 月 8 日選近十年，分析月份為 2016 年 7 月到 2026 年 6 月
- 限制：long-only，單一標的權重介於 0% 到 100%，總權重等於 100%
- 目標：最大化年化 Sharpe ratio

## 注意事項

- yfinance 是免費資料來源，可能因網路、Yahoo Finance 限流或 ticker 資料狀態而失敗。
- 第一版不支援本機 CSV 上傳、不支援放空、不支援槓桿、不支援單股權重上限。
- 最佳化結果是歷史資料回測，不代表未來績效。
