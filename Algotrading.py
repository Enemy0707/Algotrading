# streamlit_app.py
# Comprehensive backtesting front-end (equities full, futures/options scaffolding).
# Drop-in replacement. Requires: streamlit, yfinance, pandas, numpy, matplotlib, ta (optional)
#
# NOTE: Options & Futures engines are scaffolds only (UI + params). Equities/backtest is implemented.
# For production-grade options/futures you must plug in exchange-specific chains & margin rules.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io, json, traceback, math
from dataclasses import dataclass

# Optional indicator library (used if installed)
try:
    import ta
except Exception:
    ta = None

st.set_page_config(page_title="Full Backtest Website", layout="wide")
st.title("📊 Exhaustive Backtest Website — Equities implemented, Options/Futures scaffolds")

# ----------------- Utilities -----------------
def normalize_ticker(t: str) -> str:
    if not t:
        raise ValueError("Ticker required")
    tu = t.strip().upper()
    if tu in ("NIFTY", "NIFTY50"): return "^NSEI"
    if tu in ("SENSEX", "BSE", "BSESN"): return "^BSESN"
    if tu.startswith("^") or "." in t: return t.strip()
    return tu + ".NS"

@st.cache_data(ttl=3600)
def fetch_data_yf(ticker, start, end, interval='1d'):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned for ticker or incorrect interval.")
    # prefer Adj Close
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close':'Adj_Close'})
        df['Close'] = df['Adj_Close']
    if 'Open' not in df.columns:
        raise RuntimeError("Downloaded data missing Open column (required for intraday stops).")
    df = df[['Open','High','Low','Close']] if set(['High','Low']).issubset(df.columns) else df[['Open','Close']]
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(series: pd.Series):
    peak = series.cummax()
    dd = (peak - series) / peak
    return float(dd.max()) if not series.empty else 0.0

def cagr(equity_series, start_date, end_date):
    if equity_series.empty: return 0.0
    total_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    if total_years<=0: return 0.0
    return ((equity_series.iloc[-1]/equity_series.iloc[0])**(1/total_years) - 1.0) * 100.0

def sharpe_ratio(returns, rf=0.0):
    if len(returns) < 2: return 0.0
    mean = np.mean(returns) - rf
    sd = np.std(returns, ddof=1)
    if sd == 0: return 0.0
    # annualize assuming daily data (252 trading days)
    return (mean * np.sqrt(252)) / sd

# ----------------- UI: Parameter form -----------------
with st.expander("1) General (common) — required"):
    col1, col2, col3 = st.columns(3)
    with col1:
        instrument_type = st.selectbox("Instrument type", ['Equity/Index/ETF','Futures','Options'])
        raw_ticker = st.text_input("Ticker / Symbol (e.g. RELIANCE, NIFTY, BANKNIFTY)", value="ACC")
        start_date = st.date_input("Start Date", value=dt.date(2021,9,30))
        end_date   = st.date_input("End Date", value=dt.date.today())
    with col2:
        initial_capital = st.number_input("Initial Capital (₹ / $)", value=100000.0, step=1000.0, format="%.2f")
        capital_alloc_pct = st.slider("Capital Allocation per Trade (%)", min_value=1, max_value=100, value=10)/100.0
        max_open_positions = st.number_input("Max Open Positions (integer)", value=3, min_value=1, step=1)
    with col3:
        max_hold_choice = st.selectbox("Max Holding Period", ['days','intraday','expiry'])
        max_hold_days = st.number_input("Max hold (days) - used if 'days'", value=10, min_value=1, step=1)
        data_frequency = st.selectbox("Data Frequency", ['daily','1d','60m','15m','5m'], index=0)
        reinvest_profits = st.checkbox("Reinvest profits", value=True)

with st.expander("2) Entry Rules"):
    entry_method = st.selectbox("Entry Method", ['Momentum (prior-day)','MA crossover','RSI','MACD','Custom (simple)'])
    # parameters for methods
    if entry_method == 'MA crossover':
        ma_short = st.number_input("Short MA window", value=20, min_value=1)
        ma_long  = st.number_input("Long MA window", value=50, min_value=1)
    if entry_method == 'RSI':
        rsi_period = st.number_input("RSI period", value=14, min_value=2)
        rsi_enter_thr = st.number_input("RSI entry threshold (e.g. 30)", value=30)
    if entry_method == 'Momentum (prior-day)':
        prior_day_thr = st.number_input("Prior-Day Return Entry Threshold (%)", value=0.5)/100.0
    if entry_method == 'Custom (simple)':
        custom_rule = st.text_input("Custom rule (example: Close > Close.shift(1) and Close > Close.rolling(50).mean())", value="")

with st.expander("3) Exit Rules"):
    exit_method = st.selectbox("Exit Method", ['ProfitTarget+Stop','Trailing Stop','MA cross back','Time-based (EOD)'])
    profit_target = st.number_input("Profit Target (%)", value=5.0)/100.0
    stop_loss_pct = st.number_input("Stop Loss (%)", value=2.0)/100.0
    trailing_stop = st.number_input("Trailing Stop (%)", value=2.0)/100.0
    time_exit_after_days = st.number_input("Time exit after (days)", value=10, min_value=1)

with st.expander("4) Position Sizing & Leverage"):
    sizing_method = st.selectbox("Position Sizing Method", ['% of Capital','Fixed shares','Volatility adjusted (ATR)'])
    fixed_shares = st.number_input("If Fixed shares, enter quantity", value=0, min_value=0)
    leverage = st.selectbox("Leverage multiplier", [1,2,3,5], index=0)

with st.expander("5) Risk Management"):
    per_trade_stop = st.number_input("Per-Trade Stop Loss (%)", value=2.0)/100.0
    max_daily_loss_pct = st.number_input("Max Daily Loss (%)", value=5.0)/100.0
    max_overall_drawdown_pct = st.number_input("Max Overall Drawdown (%)", value=5.0)/100.0
    monthly_profit_target_pct = st.number_input("Monthly Profit Target (%)", value=5.0)/100.0
    max_consecutive_losses = st.number_input("Max Consecutive Losses Allowed", value=5, min_value=1)

with st.expander("6) Transaction Costs"):
    brokerage = st.number_input("Brokerage per trade (flat) ₹ / $", value=0.0, format="%.2f")
    slippage_pct = st.number_input("Slippage (%)", value=0.05)/100.0
    exchange_fee_pct = st.number_input("Exchange fee (%)", value=0.01)/100.0
    taxes_pct = st.number_input("Taxes (%)", value=0.0)/100.0

with st.expander("7) Options (scaffold)"):
    st.write("Options support requires option chain & IV data. This UI collects parameters but engine not implemented.")
    option_underlying = st.text_input("Underlying (for options)", value="")
    option_type = st.selectbox("Option Type", ['Call','Put','Both'])
    strike_select_method = st.selectbox("Strike selection", ['ATM','Delta','ITM','OTM'])
    expiry_rule = st.selectbox("Expiry selection", ['Weekly','Monthly','Next N'])

with st.expander("8) Futures (scaffold)"):
    st.write("Futures parameters are collected; full rollover engine not implemented in this sample.")
    fut_underlying = st.text_input("Underlying (for futures)", value="")
    lot_size = st.number_input("Lot size", value=1)
    rollover_days_before = st.number_input("Rollover days before expiry", value=2)

with st.expander("9) Performance Settings"):
    benchmark = st.text_input("Benchmark ticker (e.g. ^NSEI for NIFTY)", value="^NSEI")
    output_formats = st.multiselect("Report output formats", ['Graph','Table','CSV','JSON'], default=['Graph','CSV','JSON'])
    metrics_to_track = st.multiselect("Key metrics", ['CAGR','Sharpe Ratio','Max Drawdown','Win %','Avg Win','Avg Loss','Profit Factor'], default=['CAGR','Sharpe Ratio','Max Drawdown','Win %'])

run = st.button("Run Backtest")

# ----------------- Backtest engine (Equities) -----------------
def run_equity_backtest(cfg):
    """
    cfg: dict with all parameters
    returns: report dict, equity_df, trades_df
    """
    try:
        ticker = cfg['ticker_norm']
        df = fetch_data_yf(ticker, cfg['start'], cfg['end'], interval=cfg['freq'])
        # Add required indicator columns as needed
        df = df.copy().dropna()
        df['Close_prev'] = df['Close'].shift(1)
        # basic indicators
        if 'ma_short' in cfg and 'ma_long' in cfg:
            df['MA_short'] = df['Close'].rolling(cfg['ma_short']).mean()
            df['MA_long'] = df['Close'].rolling(cfg['ma_long']).mean()
        if cfg.get('rsi_period') and ta is not None:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=int(cfg['rsi_period']))
        # ATR (for volatility sizing)
        if ta is not None:
            try:
                atr = ta.volatility.average_true_range(df['High'] if 'High' in df.columns else df['Close'],
                                                       df['Low'] if 'Low' in df.columns else df['Close'],
                                                       df['Close'], window=14)
                df['ATR'] = atr
            except Exception:
                df['ATR'] = np.nan
        else:
            df['ATR'] = np.nan

        # engine state
        cash = float(cfg['initial_capital'])
        equity_points = []
        positions = []  # list of dicts: {entry_date, entry_price, shares, stop_price, target_price}
        trades = []
        realized_pnl_monthly = {}
        month_start_equity = {}
        consecutive_losses = 0

        allowed_min_equity = cfg['initial_capital'] * (1.0 - cfg['max_overall_drawdown_pct'])

        # helper: compute current equity given positions and price row
        def current_equity(price):
            return cash + sum([p['shares'] * price for p in positions])

        for idx, row in df.iterrows():
            today = idx.date()
            month_str = str(idx.to_period('M'))
            if month_str not in realized_pnl_monthly:
                realized_pnl_monthly[month_str] = 0.0
                month_start_equity[month_str] = current_equity(row['Close'])

            # 1) Check existing positions for stop or take-profit or time exit (approx intraday low using Low if present else min(Open,Close))
            intraday_low = row['Low'] if 'Low' in row.index else min(row['Open'], row['Close'])
            intraday_high = row['High'] if 'High' in row.index else max(row['Open'], row['Close'])

            remaining = []
            for pos in positions:
                exited = False
                exit_price = None
                reason = None
                # stop
                if intraday_low <= pos['stop_price']:
                    exit_price = pos['stop_price']
                    reason = 'stop'
                # profit target
                elif pos['target_price'] is not None and intraday_high >= pos['target_price']:
                    exit_price = pos['target_price']
                    reason = 'target'
                else:
                    # time-based
                    hold_days = (today - pos['entry_date']).days
                    if hold_days >= cfg['max_hold_days']:
                        exit_price = row['Close']
                        reason = 'time_exit'
                if exit_price is not None:
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    # costs on exit
                    cost_exit = cfg['brokerage'] + abs(exit_price*pos['shares'])* (cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                    pnl_net = pnl - cost_exit
                    cash += pos['shares'] * exit_price - cost_exit
                    trades.append({'entry_date': pos['entry_date'].isoformat(),
                                   'exit_date': today.isoformat(),
                                   'entry_price': pos['entry_price'],
                                   'exit_price': exit_price,
                                   'shares': pos['shares'],
                                   'pnl': pnl_net,
                                   'reason': reason})
                    realized_pnl_monthly[month_str] += pnl_net
                    # update consecutive losses
                    if pnl_net < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    exited = True
                if not exited:
                    remaining.append(pos)
            positions = remaining

            # 2) compute equity and check max_daily_loss (we do not implement intraday per-day loss limit strictly w/o tick data)
            equity_now = current_equity(row['Close'])
            equity_points.append({'date': idx, 'equity': equity_now})

            # 3) monthly pause
            paused = realized_pnl_monthly[month_str] >= month_start_equity[month_str] * cfg['monthly_profit_target_pct']

            # 4) stop further entries if shutdown condition
            if equity_now < allowed_min_equity:
                # emergency close all at close
                for pos in positions:
                    exit_price = row['Close']
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    cost_exit = cfg['brokerage'] + abs(exit_price*pos['shares'])*(cfg['slippage_pct']+cfg['exchange_fee_pct']+cfg['taxes_pct'])
                    pnl_net = pnl - cost_exit
                    cash += pos['shares'] * exit_price - cost_exit
                    trades.append({'entry_date': pos['entry_date'].isoformat(),
                                   'exit_date': today.isoformat(),
                                   'entry_price': pos['entry_price'],
                                   'exit_price': exit_price,
                                   'shares': pos['shares'],
                                   'pnl': pnl_net,
                                   'reason': 'emergency_close'})
                positions = []
                equity_now = current_equity(row['Close'])
                # If still below allowed -> shutdown permanently
                if equity_now < allowed_min_equity:
                    # finalize
                    break

            # 5) Entry logic (choose method)
            if not paused and len(positions) < cfg['max_open_positions'] and consecutive_losses < cfg['max_consecutive_losses']:
                signal = False
                entry_price = None
                if cfg['entry_method'] == 'Momentum (prior-day)':
                    prev_ret = (row['Close'] - row['Close_prev'])/row['Close_prev'] if not pd.isna(row.get('Close_prev', np.nan)) else np.nan
                    if pd.notna(prev_ret) and prev_ret > cfg['prior_day_thr']:
                        signal = True
                        entry_price = row['Open']  # buy at open
                elif cfg['entry_method'] == 'MA crossover':
                    if not pd.isna(row.get('MA_short', np.nan)) and not pd.isna(row.get('MA_long', np.nan)):
                        if row['MA_short'] > row['MA_long']:
                            signal = True
                            entry_price = row['Open']
                elif cfg['entry_method'] == 'RSI':
                    if not pd.isna(row.get('RSI', np.nan)) and row['RSI'] < cfg['rsi_enter_thr']:
                        signal = True
                        entry_price = row['Open']
                elif cfg['entry_method'] == 'Custom (simple)':
                    # VERY simple and safe evaluator for pandas expressions
                    try:
                        local_df = df.loc[:idx].copy()
                        cond = eval(cfg['custom_rule'], {"df": local_df, "np": np, "pd": pd})
                        # cond may be Series; we interpret last row
                        if isinstance(cond, pd.Series):
                            cond_val = bool(cond.iloc[-1])
                        else:
                            cond_val = bool(cond)
                        if cond_val:
                            signal = True
                            entry_price = row['Open']
                    except Exception:
                        signal = False

                # if signal, compute sizing
                if signal and entry_price is not None:
                    if cfg['sizing_method'] == '% of Capital':
                        capital_for_trade = cfg['capital_alloc_pct'] * (cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))
                        # risk distance
                        stop_price = entry_price * (1.0 - cfg['per_trade_stop'])
                        stop_distance = max(1e-6, entry_price - stop_price)
                        risk_amount = capital_for_trade * cfg['per_trade_stop']  # approximate
                        shares = int((capital_for_trade) // entry_price)
                    elif cfg['sizing_method'] == 'Fixed shares':
                        shares = int(cfg['fixed_shares'])
                        stop_price = entry_price * (1.0 - cfg['per_trade_stop'])
                    elif cfg['sizing_method'] == 'Volatility adjusted (ATR)':
                        atr = row.get('ATR', np.nan)
                        if pd.notna(atr) and atr>0:
                            risk_amount = cfg['capital_alloc_pct'] * (cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))
                            # target risk per trade = capital * per_trade_stop
                            stop_distance = max(atr, entry_price*cfg['per_trade_stop'])
                            shares = int(risk_amount // (stop_distance*1.0))
                            stop_price = entry_price - stop_distance
                        else:
                            shares = 0
                    else:
                        shares = 0

                    if shares > 0:
                        # pre-entry overall-drawdown check: if stop hit immediately, would equity go below allowed_min?
                        hypothetical_cash = cash - (shares * entry_price)  # after paying entry
                        worst_equity_if_stop = hypothetical_cash + (shares * stop_price) + sum([p['shares']*row['Close'] for p in positions])
                        if worst_equity_if_stop < allowed_min_equity:
                            # skip entry and note
                            trades.append({'entry_date': today.isoformat(),
                                           'exit_date': None,
                                           'entry_price': entry_price,
                                           'exit_price': None,
                                           'shares': 0,
                                           'pnl': 0.0,
                                           'reason': 'entry_skipped_would_breach_overall_drawdown'})
                        else:
                            # execute entry: deduct cost and append position
                            cost_entry = cfg['brokerage'] + abs(entry_price*shares)*(cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                            cash -= (shares * entry_price) + cost_entry
                            positions.append({'entry_date': today, 'entry_price': entry_price, 'shares': shares,
                                              'stop_price': stop_price, 'target_price': entry_price*(1+cfg['profit_target'])})
                            trades.append({'entry_date': today.isoformat(),
                                           'exit_date': None,
                                           'entry_price': entry_price,
                                           'exit_price': None,
                                           'shares': shares,
                                           'pnl': None,
                                           'reason': 'entry'})
            # end day loop

        # build outputs
        eq_df = pd.DataFrame(equity_points)
        if not eq_df.empty:
            eq_df = eq_df.set_index('date')
        trades_df = pd.DataFrame(trades)

        # metrics
        equity_series = eq_df['equity'] if not eq_df.empty else pd.Series([cfg['initial_capital']])
        final_equity = float(equity_series.iloc[-1])
        total_return = (final_equity / cfg['initial_capital'] - 1.0)*100.0
        max_dd = compute_max_drawdown(equity_series)
        cagr_val = cagr(equity_series, cfg['start'], cfg['end'])
        # daily returns from equity series
        daily_ret = equity_series.pct_change().dropna()
        sharpe = sharpe_ratio(daily_ret.values) if not daily_ret.empty else 0.0
        # trades performance
        wins = trades_df[trades_df['pnl']>0] if not trades_df.empty else pd.DataFrame()
        losses = trades_df[trades_df['pnl']<0] if not trades_df.empty else pd.DataFrame()
        win_pct = (len(wins)/len(trades_df)*100) if len(trades_df)>0 else 0.0
        avg_win = wins['pnl'].mean() if not wins.empty else 0.0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
        profit_factor = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if (not losses.empty and losses['pnl'].sum()!=0) else np.nan

        report = {
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_dd,
            'cagr_pct': cagr_val,
            'sharpe': sharpe,
            'win_pct': win_pct,
            'avg_win': float(avg_win) if not pd.isna(avg_win) else 0.0,
            'avg_loss': float(avg_loss) if not pd.isna(avg_loss) else 0.0,
            'profit_factor': float(profit_factor) if not pd.isna(profit_factor) else None,
            'equity_series_points': eq_df.reset_index().to_dict('records'),
            'trades': trades_df.to_dict('records')
        }
        return report, eq_df.reset_index(), trades_df
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}, None, None

# ----------------- Run / UI results -----------------
if run:
    try:
        ticker_norm = normalize_ticker(raw_ticker)
    except Exception as e:
        st.error(f"Ticker normalization error: {e}")
        st.stop()

    st.info(f"Running backtest for {ticker_norm} [{instrument_type}] from {start_date} to {end_date} at '{data_frequency}' interval")

    # build cfg dict
    cfg = {
        'ticker_raw': raw_ticker,
        'ticker_norm': ticker_norm,
        'start': str(start_date),
        'end': str(end_date),
        'freq': data_frequency,
        'initial_capital': float(initial_capital),
        'capital_alloc_pct': float(capital_alloc_pct),
        'max_open_positions': int(max_open_positions),
        'max_hold_days': int(max_hold_days),
        'entry_method': entry_method,
        'ma_short': locals().get('ma_short', None),
        'ma_long': locals().get('ma_long', None),
        'rsi_period': locals().get('rsi_period', None),
        'rsi_enter_thr': locals().get('rsi_enter_thr', None),
        'prior_day_thr': locals().get('prior_day_thr', None),
        'custom_rule': locals().get('custom_rule', ''),
        'sizing_method': sizing_method,
        'fixed_shares': int(fixed_shares),
        'leverage': leverage,
        'per_trade_stop': per_trade_stop,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_overall_drawdown_pct': max_overall_drawdown_pct,
        'monthly_profit_target_pct': monthly_profit_target_pct,
        'max_consecutive_losses': int(max_consecutive_losses),
        'brokerage': float(brokerage),
        'slippage_pct': float(slippage_pct),
        'exchange_fee_pct': float(exchange_fee_pct),
        'taxes_pct': float(taxes_pct),
        'profit_target': float(profit_target),
        'trailing_stop': float(trailing_stop),
        'entry_threshold': float(entry_threshold),
        'capital_alloc_pct': float(capital_alloc_pct),
        'reinvest_profits': bool(reinvest_profits),
    }

    if instrument_type != 'Equity/Index/ETF':
        st.warning("Note: Futures & Options engines are scaffolds only in this version. Equities/backtester will run now.")
    # run equities engine
    result, eq_df, trades_df = run_equity_backtest(cfg)
    if result is None:
        st.error("Backtest returned no result")
        st.stop()
    if 'error' in result:
        st.error("Backtest Error:")
        st.text(result['error'])
        st.text(result['traceback'])
        st.stop()

    # Show summary metrics
    st.subheader("Summary Metrics")
    cols = st.columns(4)
    cols[0].metric("Final equity", f"{result['final_equity']:.2f}")
    cols[1].metric("Total return %", f"{result['total_return_pct']:.2f}%")
    cols[2].metric("Max drawdown", f"{result['max_drawdown_pct']:.2%}")
    cols[3].metric("CAGR %", f"{result['cagr_pct']:.2f}%")

    # Additional metrics
    st.write("Sharpe:", f"{result['sharpe']:.3f}", "Win%:", f"{result['win_pct']:.2f}", "Profit Factor:", result.get('profit_factor'))

    # Plot equity
    if eq_df is not None and not eq_df.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(pd.to_datetime(eq_df['date']), eq_df['equity'], label='Equity')
        ax.axhline(cfg['initial_capital']*(1-cfg['max_overall_drawdown_pct']), color='red', linestyle='--', label='Allowed min equity')
        ax.set_title(f"Equity Curve: {ticker_norm}")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity")
        ax.legend()
        st.pyplot(fig)

    # Trades table & downloads
    st.subheader("Trades")
    if trades_df is None or trades_df.empty:
        st.write("No trades executed.")
    else:
        st.dataframe(trades_df)
        buf = io.StringIO()
        trades_df.to_csv(buf, index=False)
        st.download_button("Download trades.csv", buf.getvalue().encode(), file_name="trades.csv")

    # Equity CSV
    if eq_df is not None and not eq_df.empty:
        buf2 = io.StringIO()
        eq_df.to_csv(buf2, index=False)
        st.download_button("Download equity_curve.csv", buf2.getvalue().encode(), file_name="equity_curve.csv")

    # JSON report
    st.subheader("Report JSON")
    st.json(result)

    # Benchmark overlay (optional)
    if benchmark:
        try:
            bench_df = fetch_data_yf(benchmark, cfg['start'], cfg['end'], interval='1d')
            bench_price = bench_df['Close']
            # normalize to initial capital
            bench_eq = bench_price / bench_price.iloc[0] * cfg['initial_capital']
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(pd.to_datetime(eq_df['date']), eq_df['equity'], label='Strategy')
            ax2.plot(bench_eq.index, bench_eq.values, label=f'Benchmark {benchmark}')
            ax2.legend(); ax2.set_title("Strategy vs Benchmark")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Benchmark fetch failed: {e}")

    st.success("Backtest complete")

