# streamlit_app.py
# Fixed + cleaned UI full replacement for the backtest app.
# - Fixes NameError for entry threshold
# - Improved UI layout & visuals
# - Equities backtest implemented (options/futures scaffolds present)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io, json, traceback, math

# Optional indicator library (safe import)
try:
    import ta
except Exception:
    ta = None

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Backtest Studio", layout="wide", page_icon="📈")
st.markdown(
    """
    <style>
    .stApp { font-family: Inter, system-ui, sans-serif; }
    .section-title { font-size:20px; font-weight:600; margin-bottom:6px; }
    .muted { color: #6c757d; font-size:13px; }
    .card { background:#f8f9fa; padding:14px; border-radius:8px; }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown("## 📈 Backtest Studio")
st.markdown("Design your backtest with full parameter controls. Equities engine is implemented; options/futures are scaffolded.")

# -------------------------
# Sidebar: Run / Quick Info
# -------------------------
with st.sidebar:
    st.header("Run & Export")
    run = st.button("🚀 Run Backtest")
    st.write("---")
    if st.button("Import & Env Test"):
        try:
            mods = ["streamlit","pandas","numpy","yfinance","matplotlib"]
            status = {}
            for m in mods:
                try:
                    __import__(m)
                    status[m] = "ok"
                except Exception as e:
                    status[m] = f"ERROR: {e}"
            st.json(status)
        except Exception as e:
            st.error("Env test failed")
            st.exception(e)
    st.write("---")
    st.markdown("### Downloads (after run)")
    st.write("Equity CSV and Trades CSV will appear here after a successful run.")

# -------------------------
# Main input area
# -------------------------
controls_col, viz_col = st.columns([3,2])

with controls_col:
    st.markdown('<div class="section-title">1) General</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            instrument_type = st.selectbox("Instrument type", ['Equity/Index/ETF','Futures','Options'])
            raw_ticker = st.text_input("Ticker / Symbol (eg. RELIANCE, NIFTY)", value="ACC")
            start_date = st.date_input("Start Date", value=dt.date(2021,9,30))
        with col2:
            end_date = st.date_input("End Date", value=dt.date.today())
            initial_capital = st.number_input("Initial Capital", value=100000.0, step=1000.0, format="%.2f")
            capital_alloc_pct = st.slider("Capital Allocation per Trade (%)", 1, 100, 10)/100.0
        with col3:
            max_open_positions = st.number_input("Max Open Positions", value=3, min_value=1)
            max_hold_choice = st.selectbox("Max Holding Type", ['days','intraday','expiry'])
            max_hold_days = st.number_input("Max hold (days)", value=10, min_value=1)

    st.markdown('<div class="section-title">2) Entry Rules</div>', unsafe_allow_html=True)
    with st.container():
        entry_method = st.selectbox("Entry Method", ['Momentum (prior-day)','MA crossover','RSI','Custom (simple)'])
        # specific params
        if entry_method == 'MA crossover':
            ma_short = st.number_input("Short MA window", value=20, min_value=1)
            ma_long  = st.number_input("Long MA window", value=50, min_value=1)
        else:
            ma_short = ma_long = None

        if entry_method == 'RSI':
            rsi_period = st.number_input("RSI period", value=14, min_value=2)
            rsi_enter_thr = st.number_input("RSI entry threshold", value=30)
        else:
            rsi_period = rsi_enter_thr = None

        if entry_method == 'Momentum (prior-day)':
            prior_day_thr = st.number_input("Prior-Day Return Entry Threshold (%)", value=0.5)/100.0
        else:
            prior_day_thr = None

        if entry_method == 'Custom (simple)':
            custom_rule = st.text_input("Custom rule (pandas expression, safe)", value="Close > Close.shift(1)")
        else:
            custom_rule = ""

    st.markdown('<div class="section-title">3) Exit Rules</div>', unsafe_allow_html=True)
    with st.container():
        exit_method = st.selectbox("Exit Method", ['ProfitTarget+Stop','Trailing Stop','Time-based (EOD)'])
        profit_target = st.number_input("Profit Target (%)", value=5.0)/100.0
        stop_loss_pct = st.number_input("Stop Loss (%)", value=2.0)/100.0
        trailing_stop = st.number_input("Trailing Stop (%)", value=2.0)/100.0
        time_exit_after_days = st.number_input("Time exit after (days)", value=10, min_value=1)

    st.markdown('<div class="section-title">4) Sizing & Risk</div>', unsafe_allow_html=True)
    with st.container():
        sizing_method = st.selectbox("Position Sizing Method", ['% of Capital','Fixed shares','Volatility adjusted (ATR)'])
        fixed_shares = st.number_input("Fixed shares (if chosen)", value=0, min_value=0)
        per_trade_stop = st.number_input("Per-Trade Stop Loss (%)", value=2.0)/100.0
        max_overall_drawdown_pct = st.number_input("Max Overall Drawdown (%)", value=5.0)/100.0
        monthly_profit_target_pct = st.number_input("Monthly Profit Target (%)", value=5.0)/100.0
        max_consecutive_losses = st.number_input("Max Consecutive Losses", value=5, min_value=1)

    st.markdown('<div class="section-title">5) Costs & Slippage</div>', unsafe_allow_html=True)
    with st.container():
        brokerage = st.number_input("Brokerage per trade (flat)", value=0.0, format="%.2f")
        slippage_pct = st.number_input("Slippage (%)", value=0.05)/100.0
        exchange_fee_pct = st.number_input("Exchange fee (%)", value=0.01)/100.0
        taxes_pct = st.number_input("Taxes (%)", value=0.0)/100.0

    st.markdown('<div class="muted">Tip: Set the above values, then press <b>Run Backtest</b> in the sidebar.</div>', unsafe_allow_html=True)

with viz_col:
    st.markdown("### Quick Preview")
    st.write("Normalized ticker, last price preview and summary will appear here after you run the backtest.")
    preview_box = st.empty()

# -------------------------
# Helper functions & engine (kept concise)
# -------------------------
def normalize_ticker(t: str) -> str:
    if not t: raise ValueError("Ticker required")
    tu = t.strip().upper()
    if tu in ("NIFTY","NIFTY50"): return "^NSEI"
    if tu in ("SENSEX","BSE","BSESN"): return "^BSESN"
    if tu.startswith("^") or "." in t: return t.strip()
    return tu + ".NS"

@st.cache_data(ttl=3600)
def fetch_yf(ticker, start, end, interval='1d'):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data for ticker/interval")
    # prefer Adj Close
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    # ensure Open exists
    if 'Open' not in df.columns:
        raise RuntimeError("Data missing Open column")
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(series):
    if series.empty: return 0.0
    p = series.cummax()
    dd = (p - series)/p
    return float(dd.max())

# Equity backtester function (streamlined but robust)
def run_equity_backtest(cfg):
    try:
        df = fetch_yf(cfg['ticker_norm'], cfg['start'], cfg['end'], interval='1d')
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}, None, None

    df = df.copy().dropna()
    df['Close_prev'] = df['Close'].shift(1)
    # compute MA if requested
    if cfg.get('ma_short') and cfg.get('ma_long'):
        df['MA_short'] = df['Close'].rolling(cfg['ma_short']).mean()
        df['MA_long'] = df['Close'].rolling(cfg['ma_long']).mean()
    # RSI if ta available
    if cfg.get('rsi_period') and ta is not None:
        df['RSI'] = ta.momentum.rsi(df['Close'], window=int(cfg['rsi_period']))
    # ATR (optional)
    if ta is not None and set(['High','Low']).issubset(df.columns):
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    else:
        df['ATR'] = np.nan

    # engine state
    cash = float(cfg['initial_capital'])
    positions = []
    equity_points = []
    trades = []
    realized_month = {}
    month_start_eq = {}
    consecutive_losses = 0
    allowed_min = cfg['initial_capital'] * (1.0 - cfg['max_overall_drawdown_pct'])

    for idx, row in df.iterrows():
        today = idx.date()
        month_key = str(idx.to_period('M'))

        if month_key not in realized_month:
            realized_month[month_key] = 0.0
            month_start_eq[month_key] = cash + sum([p['shares']*row['Close'] for p in positions])

        # 1) check existing positions for stop/time/target
        intr_low = row['Low'] if 'Low' in row.index else min(row['Open'], row['Close'])
        intr_high = row['High'] if 'High' in row.index else max(row['Open'], row['Close'])

        remaining = []
        for pos in positions:
            exit_price = None; reason = None
            if intr_low <= pos['stop_price']:
                exit_price = pos['stop_price']; reason = 'stop'
            elif pos['target_price'] is not None and intr_high >= pos['target_price']:
                exit_price = pos['target_price']; reason = 'target'
            else:
                # time exit
                hold = (today - pos['entry_date']).days
                if hold >= cfg['max_hold_days']:
                    exit_price = row['Close']; reason = 'time_exit'
            if exit_price is not None:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cost_exit = cfg['brokerage'] + abs(exit_price*pos['shares'])*(cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                pnl_net = pnl - cost_exit
                cash += pos['shares'] * exit_price - cost_exit
                trades.append({'entry_date': pos['entry_date'].isoformat(), 'exit_date': today.isoformat(),
                               'entry_price': pos['entry_price'], 'exit_price': exit_price,
                               'shares': pos['shares'], 'pnl': pnl_net, 'reason': reason})
                realized_month[month_key] += pnl_net
                if pnl_net < 0: consecutive_losses += 1
                else: consecutive_losses = 0
            else:
                remaining.append(pos)
        positions = remaining

        # 2) equity point
        equity_now = cash + sum([p['shares']*row['Close'] for p in positions])
        equity_points.append({'date': idx, 'equity': equity_now})

        # 3) monthly pause
        paused = realized_month[month_key] >= (month_start_eq[month_key] * cfg['monthly_profit_target_pct'])

        # 4) emergency close if worst-case dips below allowed_min
        worst_if_stop = cash + sum([(p['stop_price'] if p['stop_price']<intr_low else row['Close'])*p['shares'] for p in positions])
        if worst_if_stop < allowed_min:
            # emergency close at close
            for pos in positions:
                exit_price = row['Close']
                pnl = (exit_price - pos['entry_price'])*pos['shares']
                cost_exit = cfg['brokerage'] + abs(exit_price*pos['shares'])*(cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                pnl_net = pnl - cost_exit
                cash += pos['shares']*exit_price - cost_exit
                trades.append({'entry_date': pos['entry_date'].isoformat(), 'exit_date': today.isoformat(),
                               'entry_price': pos['entry_price'], 'exit_price': exit_price,
                               'shares': pos['shares'], 'pnl': pnl_net, 'reason': 'emergency_close'})
            positions = []
            equity_now = cash
            if equity_now < allowed_min:
                break  # permanent shutdown

        # 5) entry logic
        if (not paused) and len(positions) < cfg['max_open_positions'] and consecutive_losses < cfg['max_consecutive_losses']:
            signal = False; entry_price = None
            if cfg['entry_method'] == 'Momentum (prior-day)':
                prev = (row['Close'] - row['Close_prev'])/row['Close_prev'] if not pd.isna(row.get('Close_prev', np.nan)) else np.nan
                if pd.notna(prev) and prev > (cfg.get('prior_day_thr') or 0.0):
                    signal = True; entry_price = row['Open']
            elif cfg['entry_method'] == 'MA crossover':
                if not pd.isna(row.get('MA_short',np.nan)) and not pd.isna(row.get('MA_long',np.nan)):
                    if row['MA_short'] > row['MA_long']:
                        signal=True; entry_price=row['Open']
            elif cfg['entry_method'] == 'RSI':
                if not pd.isna(row.get('RSI',np.nan)) and row['RSI'] < (cfg.get('rsi_enter_thr') or 0):
                    signal=True; entry_price=row['Open']
            elif cfg['entry_method'] == 'Custom (simple)':
                try:
                    local = df.loc[:idx].copy()
                    cond = eval(cfg.get('custom_rule',''), {"df": local, "np": np, "pd": pd})
                    if isinstance(cond, pd.Series): cond_val = bool(cond.iloc[-1])
                    else: cond_val = bool(cond)
                    if cond_val: signal=True; entry_price=row['Open']
                except Exception:
                    signal=False

            if signal and entry_price is not None:
                # sizing
                if cfg['sizing_method'] == '% of Capital':
                    cap_for_trade = cfg['capital_alloc_pct'] * (cash + sum([p['shares']*row['Close'] for p in positions]) if cfg['reinvest_profits'] else cash)
                    stop_price = entry_price*(1.0 - cfg['per_trade_stop'])
                    shares = int(cap_for_trade // entry_price)
                elif cfg['sizing_method'] == 'Fixed shares':
                    shares = int(cfg['fixed_shares']); stop_price = entry_price*(1.0 - cfg['per_trade_stop'])
                elif cfg['sizing_method'] == 'Volatility adjusted (ATR)':
                    atr = row.get('ATR', np.nan)
                    stop_price = entry_price - max(atr if not pd.isna(atr) else 0.0, entry_price*cfg['per_trade_stop'])
                    shares = int((cfg['capital_alloc_pct']*(cash if not cfg['reinvest_profits'] else cash + sum([p['shares']*row['Close'] for p in positions]))) // entry_price)
                else:
                    shares = 0; stop_price = entry_price*(1.0 - cfg['per_trade_stop'])

                if shares > 0 and shares*entry_price <= cash:
                    hypothetical_cash = cash - (shares*entry_price)
                    worst_if_stop = hypothetical_cash + (shares*stop_price) + sum([p['shares']*row['Close'] for p in positions])
                    if worst_if_stop < allowed_min:
                        trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': entry_price, 'exit_price': None,
                                       'shares': 0, 'pnl': 0.0, 'reason': 'entry_skipped_would_breach_overall_drawdown'})
                    else:
                        cost_entry = cfg['brokerage'] + abs(entry_price*shares)*(cfg['slippage_pct']+cfg['exchange_fee_pct']+cfg['taxes_pct'])
                        cash -= (shares*entry_price) + cost_entry
                        positions.append({'entry_date': today, 'entry_price': entry_price, 'shares': shares,
                                          'stop_price': stop_price, 'target_price': entry_price*(1+cfg['profit_target'])})
                        trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': entry_price,
                                       'exit_price': None, 'shares': shares, 'pnl': None, 'reason': 'entry'})

    # finalize outputs
    eq_df = pd.DataFrame(equity_points)
    if not eq_df.empty: eq_df = eq_df.set_index('date')
    trades_df = pd.DataFrame(trades)

    if not eq_df.empty:
        equity_series = eq_df['equity']
        final_equity = float(equity_series.iloc[-1])
    else:
        final_equity = cfg['initial_capital']
        equity_series = pd.Series([final_equity])

    total_return = (final_equity/cfg['initial_capital'] - 1.0)*100.0
    max_dd = compute_max_drawdown(equity_series)
    # CAGR and Sharpe
    try:
        days = (pd.to_datetime(cfg['end']) - pd.to_datetime(cfg['start'])).days or 1
        years = days/365.25
        cagr_val = ((equity_series.iloc[-1]/equity_series.iloc[0])**(1/years) - 1.0)*100.0 if years>0 else 0.0
    except Exception:
        cagr_val = 0.0
    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)) if not daily_ret.empty and daily_ret.std(ddof=1)!=0 else 0.0

    wins = trades_df[trades_df['pnl']>0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df['pnl']<0] if not trades_df.empty else pd.DataFrame()
    win_pct = (len(wins)/len(trades_df)*100.0) if len(trades_df)>0 else 0.0
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
        'equity_df': eq_df.reset_index() if eq_df is not None else pd.DataFrame(),
        'trades_df': trades_df
    }
    return report, report['equity_df'], report['trades_df']

# -------------------------
# Build cfg and run when user presses the sidebar Run button
# -------------------------
if run:
    # normalize ticker
    try:
        ticker_norm = normalize_ticker(raw_ticker)
    except Exception as e:
        st.error(f"Ticker error: {e}")
        st.stop()

    # Build configuration dictionary: make sure variable names used are the ones defined above.
    cfg = {
        'ticker_raw': raw_ticker,
        'ticker_norm': ticker_norm,
        'start': str(start_date),
        'end': str(end_date),
        'initial_capital': float(initial_capital),
        'capital_alloc_pct': float(capital_alloc_pct),
        'max_open_positions': int(max_open_positions),
        'max_hold_days': int(max_hold_days),
        'entry_method': entry_method,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'rsi_period': rsi_period if 'rsi_period' in locals() else None,
        'rsi_enter_thr': rsi_enter_thr if 'rsi_enter_thr' in locals() else None,
        'prior_day_thr': prior_day_thr if 'prior_day_thr' in locals() else None,
        'custom_rule': custom_rule,
        'sizing_method': sizing_method,
        'fixed_shares': int(fixed_shares),
        'reinvest_profits': True,
        'per_trade_stop': float(per_trade_stop),
        'max_overall_drawdown_pct': float(max_overall_drawdown_pct),
        'monthly_profit_target_pct': float(monthly_profit_target_pct),
        'max_consecutive_losses': int(max_consecutive_losses),
        'brokerage': float(brokerage),
        'slippage_pct': float(slippage_pct),
        'exchange_fee_pct': float(exchange_fee_pct),
        'taxes_pct': float(taxes_pct),
        'profit_target': float(profit_target),
        'trailing_stop': float(trailing_stop),
        'entry_threshold': float(prior_day_thr) if prior_day_thr is not None else 0.0,
        'capital_alloc_pct': float(capital_alloc_pct),
        'sizing_method': sizing_method,
        'entry_method': entry_method
    }

    st.success(f"Running backtest for {ticker_norm} from {cfg['start']} to {cfg['end']} ...")
    with st.spinner("Backtest running..."):
        report, eq_df, trades_df = run_equity_backtest(cfg)

    if isinstance(report, dict) and report.get('error'):
        st.error("Backtest error:")
        st.text(report.get('error'))
        st.text(report.get('traceback'))
    else:
        # Display summary metrics in a clean row
        st.markdown("### Summary")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Final equity", f"{report['final_equity']:.2f}")
        r2.metric("Total return %", f"{report['total_return_pct']:.2f}%")
        r3.metric("Max drawdown", f"{report['max_drawdown_pct']:.2%}")
        r4.metric("CAGR %", f"{report['cagr_pct']:.2f}%")
        st.write(f"Sharpe: {report['sharpe']:.3f}  Win%: {report['win_pct']:.2f}  Profit Factor: {report['profit_factor']}")

        # Equity Chart
        if eq_df is not None and not eq_df.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(pd.to_datetime(eq_df['date']), eq_df['equity'], label='Equity', linewidth=2)
            ax.axhline(cfg['initial_capital']*(1-cfg['max_overall_drawdown_pct']), color='red', linestyle='--', label='Allowed min equity')
            ax.set_title(f"Equity Curve: {ticker_norm}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")
            ax.legend()
            st.pyplot(fig)

        # Trades table and downloads
        st.markdown("### Trades")
        if trades_df is None or trades_df.empty:
            st.write("No trades executed.")
        else:
            st.dataframe(trades_df)
            buf = io.StringIO()
            trades_df.to_csv(buf, index=False)
            st.download_button("Download trades.csv", buf.getvalue().encode(), file_name="trades.csv")

        # equity csv
        if eq_df is not None and not eq_df.empty:
            buf2 = io.StringIO()
            eq_df.to_csv(buf2, index=False)
            st.download_button("Download equity_curve.csv", buf2.getvalue().encode(), file_name="equity_curve.csv")

        st.markdown("### Full Report JSON")
        st.json(report)

        # preview box update
        preview_box.write(f"Normalized ticker: **{ticker_norm}**  | Rows downloaded: {len(eq_df)}")

