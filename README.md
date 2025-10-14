# Advanced Stock Market Prediction & Backtesting Engine

<img width="1920" height="1080" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/b68b70b9-77cb-4c26-961e-2c0d0ecb8097" />


This repository documents the development of a sophisticated stock market prediction model, evolving from simple neural networks to a professional-grade quantitative backtesting engine. The project rigorously tests the hypothesis that a profitable trading edge can be found in historical market data.

---

## 🏆 Key Conclusion

After extensive experimentation with multiple models and features, the primary conclusion is that a consistent, profitable edge ("alpha") is **not readily found** using standard public technical indicators alone. The final, robust backtesting on real historical data demonstrates that while the model shows occasional periods of outperformance, it does not reliably beat a simple benchmark.

This result serves as a powerful practical confirmation of the **Efficient Market Hypothesis**: that all publicly available information is already priced into the market, making it incredibly difficult to find a repeatable predictive advantage.

---

## 🧠 Final Architecture: A Regime-Aware Expert System

The final and most advanced model in this project is a multi-model system designed to adapt to changing market conditions ("regimes").

1.  **The Regime Filter (The "Manager"):** A 200-day Simple Moving Average (SMA) is used to classify the market into one of two regimes:
    * **Bull Regime:** Current Price > 200-day SMA.
    * **Bear Regime:** Current Price < 200-day SMA.

2.  **The "Bull Market" Expert:** A LightGBM classifier trained *only* on historical data from bull periods. It specializes in identifying "buy" signals in an uptrend.

3.  **The "Bear Market" Expert:** A separate LightGBM classifier trained *only* on historical data from bear periods. It specializes in risk management and capital preservation, learning to be far more cautious during downtrends.

During the backtest, the "Manager" assesses the daily market regime and delegates the trading decision to the appropriate "Expert," creating a more adaptive and intelligent strategy.

---

## 🛠️ Features Engineered

The final model was trained on a set of robust technical indicators designed to capture momentum and trend strength:

* **RSI (Relative Strength Index):** 14-day and 28-day periods.
* **MACD (Moving Average Convergence Divergence):** Including the signal and histogram.
* **ADX (Average Directional Index):** To measure the strength of a trend.
* **Rolling Momentum:** Sum of log returns over 10, 20, and 60-day windows.

---



