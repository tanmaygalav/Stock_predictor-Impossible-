import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


TICKER = "AAPL"
START_DATE = "2010-01-01"


def create_features(df):
    """Creates features and defines market regimes on real data."""
    print("--- 1. Engineering features from real market data... ---")
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(1 + df['return'])
    
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    for days in [10, 20, 60]:
        df[f'momentum_{days}'] = df['log_return'].rolling(days).sum()
    
    df['target'] = np.where(df['return'].shift(-1) > 0, 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df['regime'] = np.where(df['Close'] > df['SMA_200'], 1, -1)
    
    return df


if __name__ == '__main__':
    full_df = yf.download(TICKER, start=START_DATE, auto_adjust=True)
    
    if full_df.empty:
        print(f"Failed to download data for {TICKER}. Please check your internet connection or the ticker symbol.")
        exit()
    if isinstance(full_df.columns, pd.MultiIndex):
        full_df.columns = full_df.columns.get_level_values(0)
        
    model_data_df = create_features(full_df.copy())
    
    features = [col for col in model_data_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'return', 'log_return', 'SMA_200', 'regime']]
    
    last_date = model_data_df.index[-1]
    test_start_date = last_date - pd.DateOffset(years=1)
    
    train_df = model_data_df[model_data_df.index < test_start_date]
    test_df = model_data_df[model_data_df.index >= test_start_date].copy()
    
    print(f"\n--- 2. Training the Expert System on data up to {test_start_date.date()} ---")

    bull_train_data = train_df[train_df['regime'] == 1]
    bear_train_data = train_df[train_df['regime'] == -1]

    params = {'objective': 'binary', 'metric': 'logloss', 'verbose': -1, 'class_weight': 'balanced'}
    bull_model = lgb.LGBMClassifier(**params)
    bull_model.fit(bull_train_data[features], bull_train_data['target'])
    
    bear_model = lgb.LGBMClassifier(**params)
    bear_model.fit(bear_train_data[features], bear_train_data['target'])
    print("Expert models trained on historical data.")

    print(f"\n--- 3. Running backtest on the latest year: {test_start_date.date()} to {last_date.date()} ---")
    
    predictions = []
    for i in range(len(test_df)):
        current_features = test_df[features].iloc[[i]]
        current_regime = test_df['regime'].iloc[i]
        
        pred = bull_model.predict(current_features)[0] if current_regime == 1 else bear_model.predict(current_features)[0]
        predictions.append(pred)

    test_df['ai_signal'] = predictions
    test_df['random_signal'] = np.random.randint(0, 2, len(test_df))

    test_df['ai_return'] = np.where(test_df['ai_signal'] == 1, test_df['return'], 0)
    test_df['random_return'] = np.where(test_df['random_signal'] == 1, test_df['return'], 0)
    
    initial_investment = 10000
    test_df['ai_portfolio'] = initial_investment * (1 + test_df['ai_return']).cumprod()
    test_df['random_portfolio'] = initial_investment * (1 + test_df['random_return']).cumprod()
    
    print("Backtest finished.")
    ai_accuracy = accuracy_score(test_df['target'], test_df['ai_signal'])
    print(f"Final AI Prediction Accuracy on the latest year: {ai_accuracy:.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f'Final Backtest Results for {TICKER} (Test on Latest Year)', fontsize=20)

    ax1.plot(full_df.index, full_df['Close'], label='Actual Market Price')
    ax1.plot(model_data_df.index, model_data_df['SMA_200'], label='200-Day Moving Average', color='red', linestyle='--')
    ax1.axvline(x=test_df.index[0], color='gray', linestyle=':', linewidth=2, label='Start of Test Period')
    ax1.set_title("Full Historical Price Data", fontsize=16)
    ax1.set_ylabel("Price (USD)"); ax1.grid(True); ax1.legend()
    
    ax2.plot(test_df.index, test_df['random_portfolio'], label='Random Trader PnL', color='blue')
    ax2.plot(test_df.index, test_df['ai_portfolio'], label='AI Trader PnL', color='orange')
    ax2.set_title(f"AI Expert System vs. Random Trader (Latest Year: {test_start_date.date()} to Present)", fontsize=16)
    ax2.set_xlabel("Date"); ax2.set_ylabel("Portfolio Value (USD)"); ax2.legend(); ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show()