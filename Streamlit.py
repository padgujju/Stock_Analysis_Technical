"""Interactive S&P 500 stock technical analysis dashboard."""

import datetime
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


@st.cache_data(ttl=3600)
def get_sp500_components():
    """Fetch current S&P 500 tickers and company names from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    constituents = pd.read_html(StringIO(response.text))[0]
    tickers = constituents["Symbol"].tolist()
    companies = dict(zip(constituents["Symbol"], constituents["Security"]))
    return tickers, companies


@st.cache_data(ttl=900)
def load_data(symbol, start, end):
    """Download historical OHLCV data from Yahoo Finance."""
    yahoo_symbol = symbol.replace(".", "-")
    df = yf.download(yahoo_symbol, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")


st.title("Stock Technical Analysis")
st.write("Choose an S&P 500 company and date range to view its price chart.")

try:
    available_tickers, companies = get_sp500_components()
except Exception as exc:
    st.error(f"Could not load the S&P 500 ticker list: {exc}")
    st.stop()

st.sidebar.header("Stock Parameters")
ticker = st.sidebar.selectbox(
    "Ticker",
    available_tickers,
    format_func=lambda symbol: f"{symbol} = {companies[symbol]}",
)
start_date = st.sidebar.date_input("Start date", value=datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", value=datetime.date.today())

st.sidebar.header("Technical Analysis Parameters")
volume_flag = st.sidebar.checkbox("Add volume")
with st.sidebar.expander("SMA"):
    sma_flag = st.checkbox("Add SMA")
    sma_periods = st.number_input("SMA periods", min_value=1, max_value=200, value=20, step=1)
with st.sidebar.expander("Bollinger Bands"):
    bb_flag = st.checkbox("Add Bollinger Bands")
    bb_periods = st.number_input("BB periods", min_value=1, max_value=200, value=20, step=1)
    bb_std = st.number_input("Number of standard deviations", min_value=1, max_value=4, value=2, step=1)
with st.sidebar.expander("Relative Strength Index"):
    rsi_flag = st.checkbox("Add RSI")
    rsi_periods = st.number_input("RSI periods", min_value=1, max_value=100, value=14, step=1)
    rsi_upper = st.number_input("RSI upper level", min_value=50, max_value=90, value=70, step=1)
    rsi_lower = st.number_input("RSI lower level", min_value=10, max_value=50, value=30, step=1)

if start_date > end_date:
    st.sidebar.error("The end date must be on or after the start date.")
    st.stop()

# yfinance treats the end date as exclusive, so include the selected end date.
end_exclusive = end_date + datetime.timedelta(days=1)
try:
    df = load_data(ticker, start_date, end_exclusive)
except Exception as exc:
    st.error(f"Could not download data for {ticker}: {exc}")
    st.stop()

if df.empty:
    st.warning("No price data was returned for that ticker and date range.")
    st.stop()

required_columns = ["Open", "High", "Low", "Close"]
if not all(column in df.columns for column in required_columns):
    st.error(f"Downloaded data is missing required columns: {required_columns}")
    st.stop()

df = df.dropna(subset=required_columns).copy()
if sma_flag:
    df["SMA"] = df["Close"].rolling(window=int(sma_periods)).mean()

if bb_flag:
    rolling_mean = df["Close"].rolling(window=int(bb_periods)).mean()
    rolling_std = df["Close"].rolling(window=int(bb_periods)).std()
    df["BB Upper"] = rolling_mean + int(bb_std) * rolling_std
    df["BB Lower"] = rolling_mean - int(bb_std) * rolling_std

if rsi_flag:
    delta = df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / int(rsi_periods), min_periods=int(rsi_periods), adjust=False
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / int(rsi_periods), min_periods=int(rsi_periods), adjust=False
    ).mean()
    relative_strength = average_gain / average_loss
    df["RSI"] = 100 - (100 / (1 + relative_strength))
    df.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100
    df.loc[(average_loss == 0) & (average_gain == 0), "RSI"] = 50

with st.expander("Preview and download data"):
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Columns to show", available_columns, default=available_columns
    )
    st.dataframe(df[selected_columns], width="stretch")
    csv_data = convert_df_to_csv(df[selected_columns])
    st.download_button(
        label="Download selected data as CSV",
        data=csv_data,
        file_name=f"{ticker}_stock_prices.csv",
        mime="text/csv",
    )

# Price always gets one row; volume and RSI each get their own optional row.
rows = 1 + int(volume_flag) + int(rsi_flag)
if volume_flag and rsi_flag:
    row_heights = [0.6, 0.2, 0.2]
elif volume_flag or rsi_flag:
    row_heights = [0.7, 0.3]
else:
    row_heights = [1]

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
)
price_row = 1
volume_row = 2 if volume_flag else None
rsi_row = 3 if volume_flag and rsi_flag else (2 if rsi_flag else None)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ),
    row=price_row,
    col=1,
)
if sma_flag:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA"], name=f"SMA {int(sma_periods)}"),
        row=price_row,
        col=1,
    )
if bb_flag:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB Upper"], name="BB Upper", line=dict(dash="dash")),
        row=price_row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB Lower"], name="BB Lower", line=dict(dash="dash")),
        row=price_row,
        col=1,
    )
if volume_flag and "Volume" in df.columns:
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume"),
        row=volume_row,
        col=1,
    )
if rsi_flag:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI"),
        row=rsi_row,
        col=1,
    )
    fig.add_hline(y=rsi_upper, line_dash="dash", row=rsi_row, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", row=rsi_row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=rsi_row, col=1)

fig.update_layout(
    title=f"{companies[ticker]} ({ticker}) stock price",
    xaxis_rangeslider_visible=False,
    height=800 if rows > 1 else 600,
)
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, width="stretch", key="stock_chart")
