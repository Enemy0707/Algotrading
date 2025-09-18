# streamlit_app.py
# Dark-mode backtest app with enforced white fonts on black background.
# Full replacement — overwrite existing file and redeploy.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io, traceback

# optional ta import
try:
    import ta
except Exception:
    ta = None

# ---- Page config + strict dark CSS (white text) ----
st.set_page_config(page_title="Backtest Studio (Dark, White Text)", layout="wide", page_icon="📈")
st.markdown(
    """
    <style>
    /* Background and base text */
    html, body, .stApp, .block-container {
      background: #000000 !important;
      color: #FFFFFF !important;
    }
    /* Headings and section titles */
    .section-title { font-size:18px; font-weight:600; color:#FFFFFF; margin-bottom:6px; }
    /* Muted text */
    .muted { color: #BFC7CC; font-size:13px; }
    /* Card background */
    .card { background:#0b0b0b; padding:12px; border-radius:8px; border:1px solid #222; color:#FFFFFF; }
    /* Buttons */
    .stButton>button, .stDownloadButton>button {
      background-color:#1f2937 !important;
      color:#FFFFFF !important;
      border-radius:6px !important;
    }
    /* Streamlit metrics & small text */
    .stMetric > div { color: #FFFFFF !important; }
    .stDownloadButton>button:hover { background-color:#272f3a !important; }
    /* Dataframe/table header color */
    .stDataFrame table thead th {
      color: #FFFFFF !important;
      background: #0f1720 !important;
    }
    /* Make links readable */
    a { color: #7FDBFF !important; }
    /* Ensure code blocks are readable */
    pre, code { color: #E6E6E6 !important; background: #0b0b0b !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# set matplotlib dark background and specify text color explicitly
plt.style.use('dark_background')

st.title("📈 Backtest Studio — Dark Mode (white text)")

# ---- Sidebar (Run + quick env) ----
with st.sidebar:
    st.header("Run")
    run = st.button("🚀 Run Backtest")
    st.write("---")
    if st.button("Import & Env Test"):
        mods = ["streamlit", "pandas", "numpy", "yfinance", "matplotlib"]
        status = {}
        for m in mods:
            try:
                __import__(m)
                status[m] = "ok"
            except Exception as e:
                status[m] = f"ERROR: {e}"
        st.json(status)
    st.write("---")
    st.markdown("<div class='muted'>Dark theme with enforced white text. Use daily data for stable runs.</div>", unsafe_allow_html=True)

# ---- Main Inputs ----
col_left, col_right = st.columns([3,1])

with col_left:
    st.markdown('<div class="section-title">General</div>', unsafe_allow_html=True)
    raw_ticker = st.text_input("Ticker / Symbol (e.g. ACC, RELIANCE, NIFTY)", value="ACC")
    start_date = st.date_input("Start Date", value=dt.date(2021,9,30))
    end_date = st.date_input("End Date", value=dt.date.today())
    initial_capital = st.number_input("Initial Capital", value=100000.0, format="%.2f")
    capital_alloc_pct = st.slider("Capital Allocation per Trade (%)", 1, 100, 10)/100.0
    max_open_positions = st.number_input("Max Open Positions", value=3, min_value=1)
    max_hold_days = st.number_input("Max Hold Days", value=10, min_value=1)

    st.markdown('<div class="section-title">Entry Rules</div>', unsafe_allow_html=True)
    entry_method = st.selectbox("Entry Method", ['Momentum (prior-day)','MA crossover','RSI','Custom (simple)'])
    ma_short = ma_long = None
    rsi_period = rsi_enter_thr = None
    prior_day_thr = 0.005
    custom_rule = ""
    if entry_method == 'MA crossover':
        ma_short = st.number_input("Short MA window", value=20, min_value=1)
        ma_long = st.number_input("Long MA window", value=50, min_value=1)
    elif entry_method == 'RSI':
        rsi_period = st.number_input("RSI period", value=14, min_value=2)
        rsi_enter_thr = st.number_input("RSI entry threshold", value=30)
    elif entry_method == 'Momentum (prior-day)':
        prior_day_thr = st.number_input("Prior-Day Return Entry Threshold (%)", value=0.5)/100.0
    elif entry_method == 'Custom (simple)':
        custom_rule = st.text_input("Custom rule (example: df['Close']>df['Close'].shift(1))", value="df['Close']>df['Close'].shift(1)")

    st.markdown('<div class="section-title">Exit & Risk</div>', unsafe_allow_html=True)
    profit_target = st.number_input("Profit Target (%)", value=5.0)/100.0
    per_trade_stop = st.number_input("Per-Trade Stop Loss (%)", value=2.0)/100.0
    monthly_target_pct = st.number_input("Monthly Profit Target (%)", value=5.0)/100.0
    max_overall_drawdown_pct = st.number_input("Max Overall Drawdown (%)", value=5.0)/100.0
    max_consecutive_losses = st.number_input("Max Consecutive Losses", value=5, min_value=1)

    st.markdown('<div class="section-title">Sizing & Costs</div>', unsafe_allow_html=True)
    sizing_method = st.selectbox("Sizing Method", ['% of Capital','Fixed shares','Volatility (ATR)'])
    fixed_shares = st.number_input("Fixed shares (if used)", value=0, min_value=0)
    brokerage = st.number_input("Brokerage per trade (flat)", value=0.0, format="%.2f")
    slippage_pct = st.number_input("Slippage (%)", value=0.05)/100.0
    exchange_fee_pct = st.number_input("Exchange fee (%)", value=0.01)/100.0
    taxes_pct = st.number_input("Taxes (%)", value=0.0)/100.0

with col_right:
    st.markdown('<div class="section-title">Quick Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Set parameters on the left, then click <b>Run Backtest</b> in the sidebar.</div>', unsafe_allow_html=True)
    preview = st.empty()

# ---- Helpers ----
def normalize_ticker(t):
    if not t: raise ValueError("Ticker required")
    tu = t.strip().upper()
    if tu in ("NIFTY","NIFTY50"): return "^NSEI"
    if tu in ("SENSEX","BSE","BSESN"): return "^BSESN"
    if "." in t or t.startswith("^"): return t.strip()
    return tu + ".NS"

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned for ticker/date range.")
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Open' not in df.columns:
        raise RuntimeError("Downloaded data missing 'Open' column.")
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(series):
    if series.empty: return 0.0
    peak = series.cummax()
    dd = (peak - series) / peak
    return float(dd.max())

# ---- Backtest engine (equity) ----
def run_backtest_engine(cfg):
    try:
        df = fetch_data(cfg['ticker_norm'], cfg['start'], cfg['end'])
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}, None, None

    df = df.copy().dropna()
    # vectorized prior-day returns safe
    df['prior_ret'] = df['Close'].pct_change().shift(1)
    df['Close_prev'] = df['Close'].shift(1)

    # optional indicators
    if cfg.get('ma_short') and cfg.get('ma_long'):
        df['MA_short'] = df['Close'].rolling(int(cfg['ma_short'])).mean()
        df['MA_long']  = df['Close'].rolling(int(cfg['ma_long'])).mean()
    if cfg.get('rsi_period') and ta is not None:
        df['RSI'] = ta.momentum.rsi(df['Close'], window=int(cfg['rsi_period']))
    if ta is not None and set(['High','Low']).issubset(df.columns):
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    else:
        df['ATR'] = np.nan

    cash = float(cfg['initial_capital'])
    positions = []
    equity_points = []
    trades = []
    monthly_realized = {}
    month_start_equity = {}
    consecutive_losses = 0
    allowed_min_equity = cfg['initial_capital'] * (1.0 - cfg['max_overall_drawdown_pct'])

    for idx, row in df.iterrows():
        today = idx.date()
        month_key = str(idx.to_period('M'))

        if month_key not in monthly_realized:
            monthly_realized[month_key] = 0.0
            month_start_equity[month_key] = cash + sum([p['shares']*row['Close'] for p in positions])

        # check existing positions
        intr_low = row['Low'] if 'Low' in row.index else min(row['Open'], row['Close'])
        intr_high = row['High'] if 'High' in row.index else max(row['Open'], row['Close'])
        remaining = []
        for pos in positions:
            exit_price = None; reason = None
            if intr_low <= pos['stop_price']:
                exit_price = pos['stop_price']; reason = 'stop'
            elif pos.get('target_price') is not None and intr_high >= pos['target_price']:
                exit_price = pos['target_price']; reason = 'target'
            else:
                hold_days = (today - pos['entry_date']).days
                if hold_days >= cfg['max_hold_days']:
                    exit_price = row['Close']; reason = 'time_exit'
            if exit_price is not None:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cost_exit = cfg['brokerage'] + abs(exit_price*pos['shares'])*(cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                pnl_net = pnl - cost_exit
                cash += pos['shares'] * exit_price - cost_exit
                trades.append({'entry_date': pos['entry_date'].isoformat(),
                               'exit_date': today.isoformat(),
                               'entry_price': pos['entry_price'],
                               'exit_price': exit_price,
                               'shares': pos['shares'],
                               'pnl': pnl_net,
                               'reason': reason})
                monthly_realized[month_key] += pnl_net
                if pnl_net < 0: consecutive_losses += 1
                else: consecutive_losses = 0
            else:
                remaining.append(pos)
        positions = remaining

        # equity point
        equity_now = cash + sum([p['shares'] * row['Close'] for p in positions])
        equity_points.append({'date': idx, 'equity': equity_now})

        # monthly pause
        paused = monthly_realized[month_key] >= (month_start_equity[month_key] * cfg['monthly_target_pct'])

        # emergency check
        worst_equity = cash + sum([(p['stop_price'] if p['stop_price'] < intr_low else row['Close']) * p['shares'] for p in positions])
        if worst_equity < allowed_min_equity:
            # emergency close
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
            equity_now = cash
            if equity_now < allowed_min_equity:
                break

        # entry logic (scalar-safe)
        if (not paused) and len(positions) < cfg['max_open_positions'] and consecutive_losses < cfg['max_consecutive_losses']:
            signal = False; entry_price = None
            if cfg['entry_method'] == 'Momentum (prior-day)':
                pr = row.get('prior_ret', np.nan)
                if pd.notna(pr) and float(pr) > float(cfg.get('prior_day_thr', 0.0)):
                    signal = True; entry_price = row['Open']
            elif cfg['entry_method'] == 'MA crossover':
                if pd.notna(row.get('MA_short', np.nan)) and pd.notna(row.get('MA_long', np.nan)):
                    if float(row['MA_short']) > float(row['MA_long']):
                        signal = True; entry_price = row['Open']
            elif cfg['entry_method'] == 'RSI':
                rsi = row.get('RSI', np.nan)
                if pd.notna(rsi) and float(rsi) < float(cfg.get('rsi_enter_thr', 0)):
                    signal = True; entry_price = row['Open']
            elif cfg['entry_method'] == 'Custom (simple)':
                try:
                    local = df.loc[:idx].copy()
                    cond = eval(cfg.get('custom_rule',''), {"df": local, "np": np, "pd": pd})
                    if isinstance(cond, pd.Series):
                        cond_val = bool(cond.iloc[-1])
                    else:
                        cond_val = bool(cond)
                    if cond_val:
                        signal = True; entry_price = row['Open']
                except Exception:
                    signal = False

            if signal and entry_price is not None:
                if cfg['sizing_method'] == '% of Capital':
                    cap_for_trade = cfg['capital_alloc_pct'] * (cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))
                    stop_price = entry_price * (1.0 - cfg['per_trade_stop'])
                    shares = int(cap_for_trade // entry_price)
                elif cfg['sizing_method'] == 'Fixed shares':
                    shares = int(cfg['fixed_shares']); stop_price = entry_price*(1.0 - cfg['per_trade_stop'])
                elif cfg['sizing_method'] == 'Volatility (ATR)':
                    atr = row.get('ATR', np.nan)
                    stop_price = entry_price - max(atr if not pd.isna(atr) else 0.0, entry_price*cfg['per_trade_stop'])
                    shares = int((cfg['capital_alloc_pct']*(cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))) // entry_price)
                else:
                    shares = 0; stop_price = entry_price*(1.0 - cfg['per_trade_stop'])

                if shares > 0 and shares * entry_price <= cash:
                    hypothetical_cash = cash - (shares * entry_price)
                    worst_if_stop = hypothetical_cash + (shares * stop_price) + sum([p['shares']*row['Close'] for p in positions])
                    if worst_if_stop < allowed_min_equity:
                        trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': entry_price, 'exit_price': None, 'shares': 0, 'pnl': 0.0, 'reason': 'entry_skipped_would_breach_overall_drawdown'})
                    else:
                        cost_entry = cfg['brokerage'] + abs(entry_price*shares)*(cfg['slippage_pct']+cfg['exchange_fee_pct']+cfg['taxes_pct'])
                        cash -= (shares * entry_price) + cost_entry
                        positions.append({'entry_date': today, 'entry_price': entry_price, 'shares': shares, 'stop_price': stop_price, 'target_price': entry_price*(1+cfg['profit_target'])})
                        trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': entry_price, 'exit_price': None, 'shares': shares, 'pnl': None, 'reason': 'entry'})

    # finalize
    eq_df = pd.DataFrame(equity_points)
    if not eq_df.empty:
        eq_df = eq_df.set_index('date')
        equity_series = eq_df['equity']
    else:
        equity_series = pd.Series([cfg['initial_capital']])

    final_equity = float(equity_series.iloc[-1])
    total_return = (final_equity / cfg['initial_capital'] - 1.0) * 100.0
    max_dd = compute_max_drawdown(equity_series)
    try:
        days = (pd.to_datetime(cfg['end']) - pd.to_datetime(cfg['start'])).days or 1
        years = days / 365.25
        cagr_val = ((equity_series.iloc[-1] / equity_series.iloc[0])**(1/years) - 1.0) * 100.0 if years>0 else 0.0
    except Exception:
        cagr_val = 0.0
    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)) if not daily_ret.empty and daily_ret.std(ddof=1)!=0 else 0.0

    trades_df = pd.DataFrame(trades)
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

# ---- Run when sidebar Run pressed ----
if run:
    try:
        ticker_norm = normalize_ticker(raw_ticker)
    except Exception as e:
        st.error(f"Ticker error: {e}")
        st.stop()

    cfg = {
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
        'rsi_period': rsi_period,
        'rsi_enter_thr': rsi_enter_thr,
        'prior_day_thr': prior_day_thr,
        'custom_rule': custom_rule,
        'profit_target': float(profit_target),
        'per_trade_stop': float(per_trade_stop),
        'monthly_target_pct': float(monthly_target_pct),
        'max_overall_drawdown_pct': float(max_overall_drawdown_pct),
        'max_consecutive_losses': int(max_consecutive_losses),
        'sizing_method': sizing_method,
        'fixed_shares': int(fixed_shares),
        'brokerage': float(brokerage),
        'slippage_pct': float(slippage_pct),
        'exchange_fee_pct': float(exchange_fee_pct),
        'taxes_pct': float(taxes_pct),
        'reinvest_profits': True,
        'capital_alloc_pct': float(capital_alloc_pct),
    }

    st.success(f"Running backtest for {ticker_norm} ...")
    with st.spinner("Running..."):
        result, eq_df, trades_df = run_backtest_engine(cfg)

    if isinstance(result, dict) and result.get('error'):
        st.error("Backtest error (see details):")
        st.text(result.get('error'))
        st.text(result.get('traceback'))
    else:
        # metrics row (white text guaranteed by CSS)
        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final equity", f"{result['final_equity']:.2f}")
        c2.metric("Total return %", f"{result['total_return_pct']:.2f}%")
        c3.metric("Max drawdown", f"{result['max_drawdown_pct']:.2%}")
        c4.metric("CAGR %", f"{result['cagr_pct']:.2f}%")
        st.write(f"Sharpe: {result['sharpe']:.3f}  Win%: {result['win_pct']:.2f}  Profit Factor: {result['profit_factor']}")

        # Equity chart (ensure labels white)
        if eq_df is not None and not eq_df.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(pd.to_datetime(eq_df['date']), eq_df['equity'], label='Equity', linewidth=2, color='#00E5FF')
            ax.axhline(cfg['initial_capital']*(1-cfg['max_overall_drawdown_pct']), color='red', linestyle='--', label='Allowed min equity')
            ax.set_title(f"Equity Curve: {ticker_norm}", color='#FFFFFF')
            ax.set_xlabel("Date", color='#FFFFFF'); ax.set_ylabel("Equity", color='#FFFFFF')
            ax.tick_params(colors='#FFFFFF')
            ax.legend(facecolor='#0b0b0b', edgecolor='#222', labelcolor='#FFFFFF')
            st.pyplot(fig)

        # Trades + downloads
        st.markdown("### Trades")
        if trades_df is None or trades_df.empty:
            st.write("No trades executed.")
        else:
            st.dataframe(trades_df)
            buf = io.StringIO()
            trades_df.to_csv(buf, index=False)
            st.download_button("Download trades.csv", buf.getvalue().encode(), file_name="trades.csv")

        if eq_df is not None and not eq_df.empty:
            buf2 = io.StringIO()
            eq_df.to_csv(buf2, index=False)
            st.download_button("Download equity_curve.csv", buf2.getvalue().encode(), file_name="equity_curve.csv")

        st.markdown("### Report JSON")
        st.json(result)

    preview.write(f"Ticker: **{ticker_norm}** | {len(eq_df) if eq_df is not None else 0} rows downloaded (if any).")

