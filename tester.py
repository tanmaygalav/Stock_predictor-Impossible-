import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def dynamic_rsi_strategy(df):
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['regime'] = np.where(df['Close'] > df['SMA_200'], 'bull', 'bear')
    df['signal'] = 0
    
    bull_buy_condition = (df['regime'] == 'bull') & (df['RSI'].shift(1) > 40) & (df['RSI'] <= 40)
    df.loc[bull_buy_condition, 'signal'] = 1

    bear_sell_condition = (df['regime'] == 'bear') & (df['RSI'].shift(1) < 60) & (df['RSI'] >= 60)
    df.loc[bear_sell_condition, 'signal'] = -1
    
    return df

def backtest_strategy(ticker, strategy_function, start_date, end_date, initial_capital=10000):
    print(f"--- Running backtest for {strategy_function.__name__} on {ticker} ---")
    
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        print(f"No data downloaded for {ticker}. Skipping.")
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = strategy_function(df)
    df.dropna(inplace=True)
    
    if df.empty:
        print(f"Not enough data to run strategy for {ticker} after indicator calculation. Skipping.")
        return

    position = 0
    equity = [initial_capital]
    trades = []
    current_trade = {}

    for i in range(len(df) - 1):
        current_price = df['Close'].iloc[i]
        
        if position == 1 and df['regime'].iloc[i] == 'bull' and df['RSI'].iloc[i] > 75:
            position = 0
            current_trade['exit_date'] = df.index[i]
            current_trade['exit_price'] = current_price
            trades.append(current_trade)
        elif position == -1 and df['regime'].iloc[i] == 'bear' and df['RSI'].iloc[i] < 25:
            position = 0
            current_trade['exit_date'] = df.index[i]
            current_trade['exit_price'] = current_price
            trades.append(current_trade)

        if position == 0:
            if df['signal'].iloc[i] == 1:
                position = 1
                current_trade = {'entry_date': df.index[i], 'entry_price': current_price, 'type': 'Long'}
            elif df['signal'].iloc[i] == -1:
                position = -1
                current_trade = {'entry_date': df.index[i], 'entry_price': current_price, 'type': 'Short'}

        daily_return = df['Close'].iloc[i+1] / current_price - 1
        if position != 0:
            equity.append(equity[-1] * (1 + daily_return * position))
        else:
            equity.append(equity[-1])
            
    equity_s = pd.Series(equity, index=df.index[:len(equity)])
    
    total_return = (equity_s.iloc[-1] / equity_s.iloc[0] - 1) * 100
    buy_and_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    daily_returns = equity_s.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

    trade_log = pd.DataFrame(trades)
    if not trade_log.empty and 'exit_price' in trade_log.columns:
        trade_log.dropna(subset=['exit_price'], inplace=True)
        if not trade_log.empty:
            trade_log['pnl_pct'] = np.where(
                trade_log['type'] == 'Long',
                (trade_log['exit_price'] - trade_log['entry_price']) / trade_log['entry_price'],
                (trade_log['entry_price'] - trade_log['exit_price']) / trade_log['entry_price']
            )
            win_rate = (trade_log['pnl_pct'] > 0).mean() * 100 if len(trade_log) > 0 else 0
            avg_win = trade_log[trade_log['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if len(trade_log[trade_log['pnl_pct'] > 0]) > 0 else 0
            avg_loss = trade_log[trade_log['pnl_pct'] < 0]['pnl_pct'].mean() * 100 if len(trade_log[trade_log['pnl_pct'] < 0]) > 0 else 0
            profit_factor = trade_log[trade_log['pnl_pct'] > 0]['pnl_pct'].sum() / -trade_log[trade_log['pnl_pct'] < 0]['pnl_pct'].sum() if len(trade_log[trade_log['pnl_pct'] < 0]) > 0 and trade_log[trade_log['pnl_pct'] < 0]['pnl_pct'].sum() != 0 else np.inf
        else:
            win_rate, avg_win, avg_loss, profit_factor = 0, 0, 0, 0
    else:
        win_rate, avg_win, avg_loss, profit_factor = 0, 0, 0, 0

    print("\n" + "="*50)
    print(f"BACKTESTING REPORT: {strategy_function.__name__} on {ticker}")
    print("="*50)
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_and_hold_return:.2f}%")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total Trades: {len(trade_log)}")
    print("="*50 + "\n")
    
    if not trade_log.empty:
        print("--- TRADE LOG ---")
        print(trade_log.to_string())
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 2]})
    fig.suptitle(f'Backtest Results: {strategy_function.__name__} on {ticker}', fontsize=20)
    
    ax1.plot(df.index, df['Close'], label='Stock Price', linewidth=1.5)
    ax1.plot(df.index, df['SMA_200'], label='200-Day MA (Trend Filter)', color='red', linestyle='--')

    if not trade_log.empty:
        buy_signals = trade_log[trade_log['type'] == 'Long']
        sell_signals = trade_log[trade_log['type'] == 'Short']
        ax1.plot(buy_signals['entry_date'], buy_signals['entry_price'], '^', markersize=10, color='g', label='Enter Long')
        ax1.plot(sell_signals['entry_date'], sell_signals['entry_price'], 'v', markersize=10, color='r', label='Enter Short')

    ax1.set_title('Price Chart with Trades', fontsize=16)
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(75, color='darkorange', linestyle='--', alpha=0.7); ax2.axhline(25, color='darkorange', linestyle='--', alpha=0.7)
    ax2.axhline(60, color='red', linestyle='--', alpha=0.5); ax2.axhline(40, color='green', linestyle='--', alpha=0.5)
    ax2.set_title('RSI Indicator', fontsize=16); ax2.set_ylabel('RSI Value'); ax2.grid(True)

    ax3.plot(equity_s.index, equity_s, label='Strategy Equity Curve', color='orange')
    ax3.set_title('Portfolio Value Over Time', fontsize=16)
    ax3.set_xlabel('Date'); ax3.set_ylabel('Portfolio Value (USD)'); ax3.grid(True); ax3.legend()
    
    stats_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Buy & Hold Return: {buy_and_hold_return:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Profit Factor: {profit_factor:.2f}"
    )
    ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show()

if __name__ == '__main__':
    
    tickers_to_test = [
    # Precious Metals
    'GC=F',  # Gold
    'SI=F',  # Silver
    'PL=F',  # Platinum
    'PA=F',  # Palladium

    # Energy
    'CL=F',  # Crude Oil
    'NG=F',  # Natural Gas
    'RB=F',  # RBOB Gasoline
    
    # Industrial Metals
    'HG=F',  # Copper
]
    
    START_DATE = '2024-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in tickers_to_test:
        backtest_strategy(ticker, dynamic_rsi_strategy, START_DATE, END_DATE)