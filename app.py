import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf


# =========================================================
# VIN QUANT | QUANTITATIVE MARKET TERMINAL
# =========================================================

st.set_page_config(
    page_title="VIN QUANT | Terminal",
    page_icon="📊",
    layout="wide",
)

# -------------------- STYLE -------------------------------

st.markdown("""
<style>
.stApp {
    background: #090B0E;
    color: #E8EDF5;
}
[data-testid="stSidebar"] {
    background: #11151B;
    border-right: 1px solid #2A303A;
}
.brand {
    color: #F5A623;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 2px;
}
.subtitle {
    color: #8B96A6;
    font-size: 11px;
    letter-spacing: 1.5px;
}
.panel {
    background: #11161D;
    border: 1px solid #2A303A;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.section {
    color: #F5A623;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 1.4px;
    padding: 12px 0;
}
div[data-testid="stMetric"] {
    background: #11161D;
    border: 1px solid #2A303A;
    padding: 12px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- ASSET LIST ---------------------------

ASSETS = {
    "BTC / USD": ("coinbase", "BTC-USD"),
    "ETH / USD": ("coinbase", "ETH-USD"),
    "SOL / USD": ("coinbase", "SOL-USD"),
    "GOLD FUTURES": ("yahoo", "GC=F"),
    "S&P 500": ("yahoo", "^GSPC"),
    "NASDAQ 100": ("yahoo", "^NDX"),
    "US DOLLAR INDEX": ("yahoo", "DX-Y.NYB"),
    "US TREASURY ETF": ("yahoo", "TLT"),
}


# -------------------- DATA FUNCTIONS -----------------------

def clean_ohlcv(df):
    df = df.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    required = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in required):
        raise ValueError("Kolom OHLC tidak lengkap.")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[["open", "high", "low", "close", "volume"]]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    if df.empty:
        raise ValueError("Tidak ada candle valid dari sumber data.")

    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_coinbase_data(product):
    # Coinbase public candles API. Hourly bars, recent 7 days.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": 3600,
    }
    headers = {"User-Agent": "VIN-QUANT-Research/1.0"}

    response = requests.get(
        url, params=params, headers=headers, timeout=25
    )
    response.raise_for_status()
    rows = response.json()

    if not isinstance(rows, list) or not rows:
        raise ValueError("Coinbase tidak mengembalikan data candle.")

    df = pd.DataFrame(
        rows,
        columns=["epoch", "low", "high", "open", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.drop(columns=["epoch"]).set_index("timestamp")

    return clean_ohlcv(df)


@st.cache_data(ttl=300, show_spinner=False)
def get_yahoo_data(symbol, period, interval):
    # Yahoo hourly history is limited; daily supports longer periods.
    if interval == "1h":
        period = "7d"

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=25,
    )

    if df is None or df.empty:
        raise ValueError(
            f"Yahoo Finance tidak mengembalikan data untuk {symbol}."
        )

    # Handle yfinance MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    return clean_ohlcv(df)


def load_market_data(asset, period, timeframe):
    source, symbol = ASSETS[asset]

    if source == "coinbase":
        return get_coinbase_data(symbol)

    interval = "1h" if timeframe == "1H" else "1d"
    return get_yahoo_data(symbol, period, interval)


# -------------------- QUANT FEATURES -----------------------

def add_quant_features(df, fast, slow, fee_bps):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()

    df["rolling_mean"] = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["z_score"] = (
        (df["close"] - df["rolling_mean"]) /
        rolling_std.replace(0, np.nan)
    )

    periods_year = 24 * 365 if (
        df.index.to_series().diff().median() < pd.Timedelta("2D")
    ) else 252

    df["volatility"] = (
        df["log_return"].rolling(20).std() * np.sqrt(periods_year)
    )

    df["drawdown"] = df["close"] / df["close"].cummax() - 1

    # Long-or-cash SMA crossover. Position is lagged to avoid look-ahead.
    df["position"] = (
        df["sma_fast"] > df["sma_slow"]
    ).astype(int)

    df["position_lag"] = df["position"].shift(1).fillna(0)
    turnover = df["position"].diff().abs().fillna(0)
    fee = fee_bps / 10000

    df["strategy_return"] = (
        df["position_lag"] * df["return"].fillna(0)
        - turnover * fee
    )

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["return"].fillna(0)).cumprod()
    df["strategy_drawdown"] = (
        df["strategy_equity"] /
        df["strategy_equity"].cummax() - 1
    )

    return df.replace([np.inf, -np.inf], np.nan)


def get_metrics(df, timeframe):
    r = df["strategy_return"].dropna()
    if len(r) < 2:
        return {}

    periods_year = 24 * 365 if timeframe == "1H" else 252
    equity = df["strategy_equity"]
    total_return = equity.iloc[-1] - 1

    years = max(len(r) / periods_year, 1 / periods_year)
    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if equity.iloc[-1] > 0 else -1
    )

    std = r.std()
    sharpe = (
        np.sqrt(periods_year) * r.mean() / std
        if pd.notna(std) and std > 0 else np.nan
    )

    active = df["position_lag"].astype(bool)
    trade_bars = df.loc[active, "strategy_return"]
    wins = trade_bars[trade_bars > 0]
    losses = trade_bars[trade_bars < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else np.nan
    )

    return {
        "Total Return": total_return,
        "CAGR (approx.)": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": df["strategy_drawdown"].min(),
        "Win Rate (active bars)": (
            (trade_bars > 0).mean() if len(trade_bars) else np.nan
        ),
        "Profit Factor (bar-based)": profit_factor,
    }


def percent(value):
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def excel_bytes(df):
    buffer = io.BytesIO()
    export = df.copy()

    if isinstance(export.index, pd.DatetimeIndex) and export.index.tz:
        export.index = export.index.tz_localize(None)

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="VIN_QUANT")
    return buffer.getvalue()


# -------------------- SIDEBAR ------------------------------

st.sidebar.markdown(
    '<div class="brand">VIN QUANT</div>'
    '<div class="subtitle">QUANTITATIVE RESEARCH TERMINAL</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE",
    ["Market Overview", "Quant Analytics", "Backtest Lab", "Data Explorer"],
)

asset = st.sidebar.selectbox("Market", list(ASSETS.keys()))
timeframe = st.sidebar.selectbox("Timeframe", ["1H", "1D"])

if timeframe == "1H":
    period_label = st.sidebar.selectbox("Period", ["7d"])
    period = "7d"
else:
    period_label = st.sidebar.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2
    )
    period = period_label

st.sidebar.markdown("---")
fast = st.sidebar.slider("Fast SMA", 5, 50, 20)
slow = st.sidebar.slider("Slow SMA", 20, 200, 50)
capital = st.sidebar.number_input(
    "Initial capital ($)", min_value=100.0, value=10000.0, step=500.0
)
fee_bps = st.sidebar.number_input(
    "Trading cost (bps per position change)",
    min_value=0.0, max_value=100.0, value=5.0, step=1.0,
)

if fast >= slow:
    st.sidebar.warning("Fast SMA sebaiknya lebih kecil daripada Slow SMA.")

if st.sidebar.button("Refresh market data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# -------------------- HEADER --------------------------------

st.markdown(
    """
    <div class="panel">
      <div class="brand">VIN QUANT / TERMINAL</div>
      <div class="subtitle">
        MARKET DATA &nbsp; • &nbsp; QUANT ANALYTICS &nbsp; • &nbsp; STRATEGY RESEARCH
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- FETCH & RENDER ------------------------

try:
    with st.spinner("Mengambil data pasar publik..."):
        raw = load_market_data(asset, period, timeframe)

    df = add_quant_features(raw, fast, slow, fee_bps)
    last = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else last
    price_change = last["close"] / previous["close"] - 1

    st.markdown(
        '<div class="section">01 / MARKET SNAPSHOT</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LAST PRICE", f"{last['close']:,.2f}", f"{price_change:.2%}")
    c2.metric("CANDLES", f"{len(df):,}")
    c3.metric("LOG RETURN", percent(last["log_return"]))
    c4.metric("VOLATILITY 20", percent(last["volatility"]))
    c5.metric("PRICE DRAWDOWN", percent(last["drawdown"]))

    st.caption(
        f"Asset: {asset} | Source: {ASSETS[asset][0]} | "
        f"Timeframe: {timeframe} | Last candle: {df.index[-1]}"
    )

    if page in ["Market Overview", "Quant Analytics"]:
        st.markdown(
            '<div class="section">02 / PRICE STRUCTURE</div>',
            unsafe_allow_html=True,
        )

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.04, row_heights=[0.75, 0.25],
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candles",
                increasing_line_color="#22C77A",
                decreasing_line_color="#F05D5E",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["sma_fast"],
                name=f"SMA {fast}",
                line=dict(color="#F5A623", width=1.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["sma_slow"],
                name=f"SMA {slow}",
                line=dict(color="#5C9DFF", width=1.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"], name="Volume",
                marker_color="#66758B",
            ),
            row=2, col=1,
        )

        fig.update_layout(
            template="plotly_dark", height=620,
            paper_bgcolor="#090B0E", plot_bgcolor="#090B0E",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.03),
        )
        fig.update_xaxes(gridcolor="#252B34")
        fig.update_yaxes(gridcolor="#252B34")
        st.plotly_chart(fig, use_container_width=True)

    if page == "Quant Analytics":
        st.markdown(
            '<div class="section">03 / QUANTITATIVE FEATURES</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)

        with left:
            zfig = go.Figure()
            zfig.add_trace(go.Scatter(
                x=df.index, y=df["z_score"],
                name="Z-Score", line=dict(color="#B68AFF")
            ))
            zfig.add_hline(y=2, line_dash="dash", line_color="#F05D5E")
            zfig.add_hline(y=-2, line_dash="dash", line_color="#22C77A")
            zfig.update_layout(
                title="Price Z-Score", height=320,
                template="plotly_dark", paper_bgcolor="#090B0E",
                plot_bgcolor="#090B0E",
            )
            st.plotly_chart(zfig, use_container_width=True)

        with right:
            vfig = go.Figure()
            vfig.add_trace(go.Scatter(
                x=df.index, y=df["volatility"] * 100,
                name="Volatility", line=dict(color="#F5A623"),
                fill="tozeroy",
            ))
            vfig.update_layout(
                title="Annualized Rolling Volatility (%)",
                height=320, template="plotly_dark",
                paper_bgcolor="#090B0E", plot_bgcolor="#090B0E",
            )
            st.plotly_chart(vfig, use_container_width=True)

        z = last["z_score"]
        if pd.isna(z):
            st.info("Z-Score belum tersedia karena jumlah candle belum cukup.")
        elif z >= 2:
            st.info("Harga lebih dari 2 standar deviasi di atas rolling mean.")
        elif z <= -2:
            st.info("Harga lebih dari 2 standar deviasi di bawah rolling mean.")
        else:
            st.info("Harga berada di antara -2 dan +2 standar deviasi.")

        st.caption("Z-Score bukan sinyal trading mandiri dan tidak menjamin pembalikan harga.")

    if page == "Backtest Lab":
        st.markdown(
            '<div class="section">03 / STRATEGY BACKTEST</div>',
            unsafe_allow_html=True,
        )
        st.write(
            f"Aturan: long saat SMA {fast} > SMA {slow}; selain itu cash. "
            f"Biaya simulasi {fee_bps:.1f} bps per perubahan posisi."
        )

        metrics = get_metrics(df, timeframe)
        final_value = capital * df["strategy_equity"].iloc[-1]
        buy_hold_value = capital * df["buy_hold_equity"].iloc[-1]

        a, b, c, d = st.columns(4)
        a.metric("STRATEGY FINAL", f"${final_value:,.2f}")
        b.metric("BUY & HOLD FINAL", f"${buy_hold_value:,.2f}")
        c.metric("STRATEGY RETURN", percent(df["strategy_equity"].iloc[-1] - 1))
        d.metric("MAX DRAWDOWN", percent(df["strategy_drawdown"].min()))

        eq = go.Figure()
        eq.add_trace(go.Scatter(
            x=df.index, y=capital * df["strategy_equity"],
            name="SMA Strategy", line=dict(color="#F5A623", width=2),
        ))
        eq.add_trace(go.Scatter(
            x=df.index, y=capital * df["buy_hold_equity"],
            name="Buy & Hold", line=dict(color="#5C9DFF", width=1.5),
        ))
        eq.update_layout(
            title="Equity Curve", height=400,
            template="plotly_dark", paper_bgcolor="#090B0E",
            plot_bgcolor="#090B0E", hovermode="x unified",
        )
        st.plotly_chart(eq, use_container_width=True)

        if metrics:
            metric_df = pd.DataFrame(
                [{"Metric": k, "Value": percent(v) if "Return" in k or "Drawdown" in k or "Rate" in k
                  else (f"{v:.3f}" if pd.notna(v) else "N/A")}
                 for k, v in metrics.items()]
            )
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.warning(
            "Ini backtest edukasi, bukan rekomendasi investasi. Win rate dan profit factor "
            "dihitung berdasarkan bar saat posisi aktif, bukan transaksi yang dipasangkan. "
            "Hasil belum mencakup spread aktual, slippage, pajak, atau keterlambatan eksekusi."
        )

    if page == "Data Explorer":
        st.markdown(
            '<div class="section">02 / HISTORICAL DATA EXPLORER</div>',
            unsafe_allow_html=True,
        )

        st.write(f"Dataset: **{asset}** | **{timeframe}**")
        st.write(
            f"Periode candle: {df.index.min()} sampai {df.index.max()}"
        )
        st.metric("Rows", f"{len(df):,}")
        st.dataframe(
            df.sort_index(ascending=False),
            use_container_width=True,
            height=450,
        )

        csv_data = df.to_csv().encode("utf-8")
        xlsx_data = excel_bytes(df)

        left, right = st.columns(2)
        left.download_button(
            "Download CSV",
            data=csv_data,
            file_name="VIN_QUANT_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        right.download_button(
            "Download Excel",
            data=xlsx_data,
            file_name="VIN_QUANT_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "VIN QUANT | Data publik dapat terlambat atau tidak tersedia. "
        "Informasi untuk riset dan edukasi, bukan rekomendasi investasi."
    )

except Exception as error:
    st.error("Data gagal dimuat atau terjadi kesalahan saat pemrosesan.")
    st.code(f"{type(error).__name__}: {error}")
    st.info(
        "Coba pilih BTC / USD, lalu klik Refresh market data. "
        "Jika masih error, salin pesan error di atas."
    )
