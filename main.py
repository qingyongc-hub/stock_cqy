import pandas as pd
import numpy as np
# import cufflinks as cf
import yfinance as yf
#from pandas_datareader import data
from datetime import datetime
import datetime
#import matplotlib.pyplot as plt
#from pandas.tseries.frequencies import to_offset
#import mplfinance as mpf
#from mpl_finance import candlestick_ohlc
import yfinance
from html import escape
from urllib.parse import quote, urlparse
#import matplotlib.dates as mpl_dates
##from fbprophet import Prophet
import streamlit as st
from plotly import graph_objs as go
from plotly import express as px
##from fbprophet.plot import plot_plotly
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.lib import FractionalBacktest, SignalStrategy, TrailingStrategy

import codecs

import streamlit.components.v1 as stc

# import prophet
# from prophet.plot import plot_plotly



#####################################################################################

#start = "2020-06-01"
#end  = datetime.today()

#ema_10 = 10
#ema_20 = 20
#ema_50 = 50
#ema_100 = 100
ema_10 = 10
ema_20 = 20


def EMA(values, period):
    """Return an exponential moving average for backtesting.py indicators."""
    return (
        pd.Series(values)
        .ewm(span=period, min_periods=period, adjust=False)
        .mean()
        .to_numpy()
    )

#####################################################################################

st.markdown("<div style='height: 3.75rem;'></div>", unsafe_allow_html=True)
st.title('Stock App')

ticker_options = [
    "GC=F", "CL=F", "BTC-USD", "VOO", "^HSI", "^KLSE",
]
ticker_from_url = st.query_params.get("ticker", "")
start_date_from_url = st.query_params.get("start", "")
end_date_from_url = st.query_params.get("end", "")


def get_date_from_url(value, fallback):
    try:
        return datetime.date.fromisoformat(str(value)) if value else fallback
    except ValueError:
        return fallback


start_date_default = get_date_from_url(
    start_date_from_url, datetime.date(2019, 1, 1)
)
end_date_default = get_date_from_url(
    end_date_from_url, datetime.date(2026, 10, 1)
)
if (
    ticker_from_url
    and st.session_state.get("_ticker_from_url") != ticker_from_url
):
    st.session_state.ticker_selector = ticker_from_url
    st.session_state._ticker_from_url = ticker_from_url

if "ticker_selector" not in st.session_state:
    st.session_state.ticker_selector = "1155.KL"
if (
    start_date_from_url
    and st.session_state.get("_start_date_from_url") != start_date_from_url
):
    st.session_state.start_date_selector = start_date_default
    st.session_state._start_date_from_url = start_date_from_url
if (
    end_date_from_url
    and st.session_state.get("_end_date_from_url") != end_date_from_url
):
    st.session_state.end_date_selector = end_date_default
    st.session_state._end_date_from_url = end_date_from_url
if "start_date_selector" not in st.session_state:
    st.session_state.start_date_selector = start_date_default
if "end_date_selector" not in st.session_state:
    st.session_state.end_date_selector = end_date_default

user_input = st.selectbox(
    "Enter Ticker",
    options=ticker_options,
    index=None,
    accept_new_options=True,
    help="Choose a listed ticker or type any Yahoo Finance symbol.",
    key="ticker_selector",
)
st.caption(
    "Sample tickers — "
    "US: AAPL, GOOGL, MSFT  |  "
    "Malaysia: 1155.KL, 5183.KL, 5296.KL  |  "
    "Hong Kong: 0700.HK, 1810.HK, 9988.HK"
)
user_input_start = st.date_input('Enter Start Date', key="start_date_selector")
user_input_end = st.date_input('Enter End Date', key="end_date_selector")
date_range_label = (
    f"{user_input_start.strftime('%Y-%m-%d')} to "
    f"{user_input_end.strftime('%Y-%m-%d')}"
)

backtest_view = st.query_params.get("view", "Daily")
if backtest_view not in {"Summary", "Daily", "Weekly"}:
    backtest_view = "Daily"

#####################################################################################

user_ticker = (user_input or "").strip().upper()
if not user_ticker:
    st.info("Choose or type a ticker symbol to begin, for example aapl or 1113.hk or 1155.kl.")
    st.stop()

try:
    df = yf.download(
        user_ticker,
        start=user_input_start,
        end=user_input_end,
        progress=False,
    )
except Exception:
    st.warning(
        f"Unable to download market data for {user_ticker}. "
        "Please check the ticker symbol and try again."
    )
    st.stop()

if df is None or df.empty:
    st.warning(
        f"No market data was found for {user_ticker}. "
        "Please check the ticker symbol, such as 1113.HK instead of 1113."
    )
    st.stop()

# yfinance 1.3.0 returns MultiIndex columns for a single ticker by default.
# Flatten them once so Plotly, cufflinks, and the backtesting code receive
# ordinary string column names.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel("Ticker")

required_columns = {"Open", "High", "Low", "Close"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.warning(
        f"Market data for {user_ticker} is missing required price columns. "
        "Please try another ticker."
    )
    st.stop()

df = df.reset_index().dropna()
df1 = df.copy()
tickerData = yf.Ticker(user_ticker) # Get ticker data

try:
    ticker_info = tickerData.info
except Exception:
    ticker_info = {}

company_name = (
    ticker_info.get("longName")
    or ticker_info.get("shortName")
    or user_ticker
)
company_country = ticker_info.get("country") or "Information unavailable"
company_logo_url = ticker_info.get("logo_url")
company_website = ticker_info.get("website") or ""
if not company_logo_url and company_website:
    company_domain = urlparse(str(company_website)).netloc
    if company_domain:
        company_logo_url = (
            "https://www.google.com/s2/favicons?domain="
            f"{quote(company_domain, safe='')}&sz=128"
        )
company_logo_html = '<span class="stock-company-logo-fallback" aria-hidden="true">↗</span>'
if company_logo_url:
    company_logo_html = (
        f'<img class="stock-company-logo" '
        f'src="{escape(str(company_logo_url), quote=True)}" '
        f'alt="{escape(str(company_name), quote=True)} logo" '
        'onerror="this.remove()">'
    )
display_company_name = escape(str(company_name))
display_company_country = escape(str(company_country))
display_ticker = escape(str(user_ticker))
ticker_url_value = quote(user_ticker, safe="")
start_date_url_value = user_input_start.isoformat()
end_date_url_value = user_input_end.isoformat()
summary_button_class = "active" if backtest_view == "Summary" else ""
daily_button_class = "active" if backtest_view == "Daily" else ""
weekly_button_class = "active" if backtest_view == "Weekly" else ""
st.markdown(
    f"""
    <style>
    .stock-company-header {{
        position: fixed;
        top: 3.25rem;
        left: 0;
        z-index: 999999;
        box-sizing: border-box;
        width: 100%;
        padding: 0.6rem max(1rem, calc((100vw - 76rem) / 2));
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        background: rgba(14, 17, 23, 0.96);
        color: #fafafa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
    }}
    .stock-company-header-inner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}
    .stock-company-details {{
        min-width: 0;
    }}
    .stock-company-identity {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-width: 0;
    }}
    .stock-company-logo {{
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border-radius: 8px;
        background: #ffffff;
        object-fit: contain;
        padding: 3px;
    }}
    .stock-company-logo-fallback {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border: 1px solid rgba(250, 250, 250, 0.35);
        border-radius: 8px;
        background: #1f2937;
        color: #60a5fa;
        font-size: 1.45rem;
        font-weight: 700;
    }}
    .stock-company-name {{
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.25;
    }}
    .stock-company-meta {{
        margin-top: 0.2rem;
        font-size: 0.82rem;
        opacity: 0.75;
    }}
    .stock-company-header-spacer {{
        height: 3.75rem;
    }}
    .stock-view-switcher {{
        display: flex;
        gap: 0.4rem;
        flex-shrink: 0;
    }}
    .stock-view-button {{
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(250, 250, 250, 0.35);
        border-radius: 0.45rem;
        color: #fafafa !important;
        font-size: 0.86rem;
        font-weight: 600;
        text-decoration: none !important;
    }}
    .stock-view-button.active {{
        border-color: #ff4b4b;
        background: #ff4b4b;
    }}
    @media (max-width: 640px) {{
        .stock-company-header-inner {{
            align-items: flex-start;
            flex-direction: column;
            gap: 0.55rem;
        }}
        .stock-company-header-spacer {{
            height: 5.6rem;
        }}
    }}
    </style>
    <div class="stock-company-header">
        <div class="stock-company-header-inner">
            <div class="stock-company-identity">
                {company_logo_html}
                <div class="stock-company-details">
                    <div class="stock-company-name">{display_company_name}</div>
                    <div class="stock-company-meta">
                        {display_ticker} · {display_company_country} · {date_range_label}
                    </div>
                </div>
            </div>
            <div class="stock-view-switcher">
                <a class="stock-view-button {daily_button_class}" href="?view=Daily&amp;ticker={ticker_url_value}&amp;start={start_date_url_value}&amp;end={end_date_url_value}" target="_self">Daily</a>
                <a class="stock-view-button {weekly_button_class}" href="?view=Weekly&amp;ticker={ticker_url_value}&amp;start={start_date_url_value}&amp;end={end_date_url_value}" target="_self">Weekly</a>
                <a class="stock-view-button {summary_button_class}" href="?view=Summary&amp;ticker={ticker_url_value}&amp;start={start_date_url_value}&amp;end={end_date_url_value}" target="_self">Summary</a>
            </div>
        </div>
    </div>
    <div class="stock-company-header-spacer"></div>
    """,
    unsafe_allow_html=True,
)


weekly_aggregations = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
}
if "Volume" in df1.columns:
    weekly_aggregations["Volume"] = "sum"

weekly = (
    df1.set_index("Date")
       .resample("W-Fri")
       .agg(weekly_aggregations)
)

if "Volume" not in weekly.columns:
    weekly["Volume"] = 0

weekly1 = weekly                                 
weekly = weekly.reset_index()

df2 = df                                 

#####################################################################################

#CHANGE A LOT AAAAAAAAA, PUT IMPORTANT PARAMETER BEFORE THIS

try:
    hist = tickerData.history(start=user_input_start,end=user_input_end)
except Exception:
    hist = pd.DataFrame()

if "Dividends" in hist.columns and hist["Dividends"].notna().any():
    hist1 = hist[["Dividends"]].copy()
else:
    hist1 = pd.DataFrame()
#hist2 = hist[["Dividends","Stock Splits"]]

#####################################################################################

#string_logo = '<img src=%s>' % tickerData.info['logo_url']
#st.markdown(string_logo, unsafe_allow_html=True)

#string_name = tickerData.info['longName']
#st.header(string_name)

#string_summary = tickerData.info['longBusinessSummary']
#st.info(string_summary)

#####################################################################################

def display_table(data, rows=None):
    table = data.tail(rows).copy() if rows else data.copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(4)
    st.dataframe(table, width="stretch", hide_index=True)


if backtest_view == "Daily":
    st.markdown(f"### Daily Market Data ({date_range_label})")
    display_table(df[["Date", "Open", "High", "Low", "Close", "Volume"]], rows=10)

# st.subheader('Weekly Market Data')
# display_table(weekly[["Date", "Open", "High", "Low", "Close"]], rows=30)

df["ema_10"] = df.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
df["ema_20"] = df.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
df=df.dropna()

#####################################################################################

#DAILY CHART

# st.subheader(' ')
# st.subheader('Day Chart')
# def plot_raw_data():
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
# 	st.plotly_chart(fig, key="daily_line_chart")
	
# plot_raw_data()

#####################################################################################

#Dividends and stock splits

# if not hist1.empty:
#     fig3 = px.line(hist1, title="Dividends")
#     st.plotly_chart(fig3, key="dividends_chart")
# else:
#     st.info(f"No dividend data is available for {user_input.upper()} in the selected period.")

#####################################################################################

# Candlestick chart
def plot_candlestick(data, title):
    chart = data.copy().dropna(subset=["Open", "High", "Low", "Close"])
    x = chart["Date"] if "Date" in chart.columns else chart.index

    chart["ema_10"] = chart["Close"].ewm(span=ema_10, min_periods=ema_10).mean()
    chart["ema_20"] = chart["Close"].ewm(span=ema_20, min_periods=ema_20).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x,
        open=chart["Open"],
        high=chart["High"],
        low=chart["Low"],
        close=chart["Close"],
        name="Candle"
    ))
    fig.add_trace(go.Scatter(x=x, y=chart["ema_10"], name="EMA 10", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x, y=chart["ema_20"], name="EMA 20", line=dict(color="red")))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

if backtest_view == "Daily":
    fig4 = plot_candlestick(
        df,
        f"Daily Chart - {user_ticker} ({date_range_label})",
    )
    st.plotly_chart(fig4, key="daily_candlestick_chart")

#####################################################################################

if backtest_view == "Weekly":
    st.markdown(f"### Weekly Market Data ({date_range_label})")
    display_table(weekly, rows=10)

    fig5 = plot_candlestick(
        weekly,
        f"Weekly Chart - {user_ticker} ({date_range_label})",
    )
    st.plotly_chart(fig5, key="weekly_candlestick_chart")

#####################################################################################

# #Forecast

# st.subheader('Forecast')
# df[['ds','y']]=df[['Date','Adj Close']]
# model=Prophet()
# model.fit(df)
# future = model.make_future_dataframe(periods=365)
# forecast = model.predict(future)

# fig6=plot_plotly(model,forecast)
# st.plotly_chart(fig6)

#####################################################################################

#Recommendations

#st.subheader('Recommendations')
#st.write(tickerData.recommendations.tail(30))

#####################################################################################

#Strategy WEEK & DAY

weekly["ema_10"] = weekly.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
weekly["ema_20"] = weekly.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
weekly["position"] = np.where(weekly["ema_10"] > weekly["ema_20"], 1, -1 )

df["ema_10"] = df.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
df["ema_20"] = df.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
df["position"] = np.where(df["ema_10"] > df["ema_20"], 1, -1 )


weekly_position = weekly[["Date","Close","ema_10","ema_20","position"]]
daily_position = df[["Date","Close","ema_10","ema_20","position"]]

# st.subheader('Daily Position')   
# display_table(daily_position, rows=1000)

# st.subheader('Weekly Position') 
# display_table(weekly_position, rows=1000)
# st.subheader('')
# st.subheader('')




#####################################################################################

st.subheader('')
st.subheader('')

def openHTML(test_html,width=900,height=800):
    file=codecs.open(test_html,'r')
    page = file.read()
    stc.html(page,width=width,height=height)

#BackTest

class EmaCross(SignalStrategy, TrailingStrategy):
    fast_period = 10
    slow_period = 20
    
    def init(self):
        super().init()
        
        ema1 = self.I(EMA, self.data.Close, self.fast_period)
        ema2 = self.I(EMA, self.data.Close, self.slow_period)

        # Where EMA 10 crosses EMA 20 upwards. Diff gives us [-1, 0, 1].
        signal = (pd.Series(ema1) > ema2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only

        entry_size = signal * .95

        self.set_signal(entry_size = entry_size)
        
        self.set_atr_periods(20)
        self.set_trailing_sl(2) #ATR stop loss (20*2)

      




#####################################################################################

class EmaCrossLongOnly(SignalStrategy):
    """Long-only EMA crossover with an EMA crossover exit."""

    fast_period = 10
    slow_period = 20

    def init(self):
        super().init()

        ema_fast = self.I(EMA, self.data.Close, self.fast_period)
        ema_slow = self.I(EMA, self.data.Close, self.slow_period)

        above_slow = (pd.Series(ema_fast) > ema_slow).astype(int)
        crossover_signal = above_slow.diff().fillna(0)

        # Buy only when EMA 10 crosses above EMA 20.
        entry_signal = (crossover_signal > 0).astype(float) * 0.95
        # Close the long position when EMA 20 crosses above EMA 10.
        exit_signal = (crossover_signal < 0).astype(float)

        self.set_signal(
            entry_size=entry_signal,
            exit_portion=exit_signal,
        )


class EmaCross30_60(EmaCross):
    """EMA 30/60 crossover with ATR trailing stop."""

    fast_period = 30
    slow_period = 60


class EmaCrossLongOnly30_60(EmaCrossLongOnly):
    """Long-only EMA 30/60 crossover with EMA crossover exit."""

    fast_period = 30
    slow_period = 60


daily_backtest_data = df1.set_index("Date").copy()
daily_backtest_data.index = pd.to_datetime(daily_backtest_data.index)
bt_daily = FractionalBacktest(daily_backtest_data,EmaCrossLongOnly,cash=1000, commission=0.002,trade_on_close=True)
bt_daily_stats = bt_daily.run()
print("Daily backtest results:")
print(bt_daily_stats.to_string())

bt_daily_atr = FractionalBacktest(daily_backtest_data,EmaCross,cash=1000, commission=0.002,trade_on_close=True)
bt_daily_atr_stats = bt_daily_atr.run()
print("Daily EMA + ATR backtest results:")
print(bt_daily_atr_stats.to_string())

bt_daily_30_60 = FractionalBacktest(
    daily_backtest_data,
    EmaCross30_60,
    cash=1000,
    commission=0.002,
    trade_on_close=True,
)
bt_daily_30_60_stats = bt_daily_30_60.run()
print("Daily EMA 30/60 + ATR backtest results:")
print(bt_daily_30_60_stats.to_string())

bt_daily_long_30_60 = FractionalBacktest(
    daily_backtest_data,
    EmaCrossLongOnly30_60,
    cash=1000,
    commission=0.002,
    trade_on_close=True,
)
bt_daily_long_30_60_stats = bt_daily_long_30_60.run()
print("Daily EMA 30/60 Exit results:")
print(bt_daily_long_30_60_stats.to_string())


strategy_colors = {
    "Short-Term Trend + ATR Stop": "#3B82F6",
    "Long-Term Trend + ATR Stop": "#A855F7",
    "Short-Term Trend + EMA Exit": "#22C55E",
    "Long-Term Trend + EMA Exit": "#F97316",
}


def get_strategy_color(title):
    for strategy_name, color in strategy_colors.items():
        if strategy_name in title:
            return color
    return None


def display_colored_heading(title):
    color = get_strategy_color(title)
    if color is None:
        st.markdown(f"### {title}")
        return

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            margin: 1.25rem 0 0.55rem 0;
            padding: 0.25rem 0 0.25rem 0.75rem;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.25;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_strategy_legend():
    legend_items = "".join(
        f"""
        <span style="
            display: inline-block;
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.35rem 0.6rem;
            border-radius: 0.4rem;
            border: 1px solid {color};
            color: {color};
            font-size: 0.82rem;
            font-weight: 600;
        ">{strategy_name}</span>
        """
        for strategy_name, color in strategy_colors.items()
    )
    st.markdown("#### Strategy Colour Guide")
    st.markdown(legend_items, unsafe_allow_html=True)


def display_trade_log(stats, title):
    trades = stats.get("_trades")
    if trades is None or trades.empty:
        st.info(f"No completed trades are available for {title}.")
        return

    trade_log = trades.copy()
    sort_columns = [
        column for column in ("EntryTime", "ExitTime") if column in trade_log.columns
    ]
    if sort_columns:
        trade_log = trade_log.sort_values(
            by=sort_columns,
            ascending=True,
            kind="stable",
        ).reset_index(drop=True)

    column_labels = {
        "EntryTime": "Entry Date",
        "ExitTime": "Exit Date",
        "Size": "Size",
        "EntryBar": "Entry Bar",
        "ExitBar": "Exit Bar",
        "EntryPrice": "Entry Price",
        "ExitPrice": "Exit Price",
        "SL": "Stop Loss",
        "TP": "Take Profit",
        "PnL": "P&L",
        "Commission": "Commission",
        "ReturnPct": "Return %",
        "Duration": "Duration",
    }
    preferred_columns = [
        "EntryTime",
        "ExitTime",
        "Size",
        "EntryBar",
        "ExitBar",
        "EntryPrice",
        "ExitPrice",
        "SL",
        "TP",
        "PnL",
        "Commission",
        "ReturnPct",
        "Duration",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in trade_log.columns
    ]
    trade_log = trade_log[ordered_columns].rename(columns=column_labels)

    for date_column in ("Entry Date", "Exit Date"):
        if date_column in trade_log.columns:
            trade_log[date_column] = pd.to_datetime(
                trade_log[date_column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    for price_column in (
        "Entry Price",
        "Exit Price",
        "Stop Loss",
        "Take Profit",
        "P&L",
        "Commission",
    ):
        if price_column in trade_log.columns:
            trade_log[price_column] = trade_log[price_column].map(
                lambda value: f"${value:,.2f}" if pd.notna(value) else "—"
            )
    if "Return %" in trade_log.columns:
        trade_log["Return %"] = trade_log["Return %"].map(
            lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "—"
        )
    if "Size" in trade_log.columns:
        trade_log["Size"] = trade_log["Size"].map(
            lambda value: (
                f"{value:,.0f}"
                if pd.notna(value) and float(value).is_integer()
                else f"{value:,.4f}"
                if pd.notna(value)
                else "—"
            )
        )

    display_colored_heading(title)
    st.dataframe(trade_log, width="stretch", hide_index=True)


def display_special_objects(stats, title):
    equity_curve = stats.get("_equity_curve")
    display_colored_heading(f"{title} Equity Curve")
    if equity_curve is None or equity_curve.empty or "Equity" not in equity_curve:
        st.info("Equity curve is unavailable.")
        return

    st.line_chart(equity_curve[["Equity"]])

strategy_descriptions = {
    "Short-Term Trend + ATR Stop": (
        "Entry: EMA 10 crosses above EMA 20. "
        "Exit: 2 × ATR(20) trailing stop. Long-only."
    ),
    "Long-Term Trend + ATR Stop": (
        "Entry: EMA 30 crosses above EMA 60. "
        "Exit: 2 × ATR(20) trailing stop. Long-only."
    ),
    "Short-Term Trend + EMA Exit": (
        "Entry: EMA 10 crosses above EMA 20. "
        "Exit: EMA 10 crosses below EMA 20. Long-only."
    ),
    "Long-Term Trend + EMA Exit": (
        "Entry: EMA 30 crosses above EMA 60. "
        "Exit: EMA 30 crosses below EMA 60. Long-only."
    ),
}


def display_backtest_stats(stats, title):
    summary = stats.drop(
        labels=["_strategy", "_equity_curve", "_trades"],
        errors="ignore",
    )
    summary = summary.rename_axis("Metric").reset_index(name="Value")

    def format_stat_value(metric, value):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "—"
        if metric in {"Start", "End"}:
            return str(value).split(" ")[0]
        if "[%]" in metric:
            return f"{float(value):.2f}%"
        if "[$]" in metric:
            return f"${float(value):,.2f}"
        if metric == "# Trades":
            return f"{int(value):,}"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.2f}"
        return str(value)

    summary["Value"] = [
        format_stat_value(metric, value)
        for metric, value in zip(summary["Metric"], summary["Value"])
    ]
    display_colored_heading(title)
    st.caption(strategy_descriptions.get(title, ""))

    key_metrics = [
        ("Start", "Start Date"),
        ("End", "End Date"),
        ("Return [%]", "Return"),
        ("Buy & Hold Return [%]", "Buy & Hold"),
        ("Equity Final [$]", "Final Equity"),
        ("Max. Drawdown [%]", "Max Drawdown"),
        ("Sharpe Ratio", "Sharpe"),
        ("# Trades", "Trades"),
        ("Win Rate [%]", "Win Rate"),
    ]
    metric_columns = st.columns(3)
    for position, (metric_name, display_name) in enumerate(key_metrics):
        value = summary.loc[summary["Metric"] == metric_name, "Value"]
        if not value.empty:
            display_value = value.iloc[0]
            with metric_columns[position % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid rgba(128,128,128,.35);
                        border-radius: 10px;
                        padding: 10px 14px;
                        margin: 4px 0 10px 0;
                        min-height: 66px;
                    ">
                        <div style="font-size: 13px; opacity: .72;">{display_name}</div>
                        <div style="font-size: 22px; font-weight: 600; white-space: nowrap;">
                            {display_value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("All backtest metrics"):
        st.dataframe(summary, width="stretch", hide_index=True)


def display_strategy_comparison(results, title):
    metrics = [
        ("Return [%]", "Return"),
        ("Buy & Hold Return [%]", "Buy & Hold Return"),
        ("Equity Final [$]", "Final Equity"),
        ("CAGR [%]", "CAGR"),
        ("Max. Drawdown [%]", "Max Drawdown"),
        ("Sharpe Ratio", "Sharpe"),
        ("Sortino Ratio", "Sortino"),
        ("# Trades", "Trades"),
        ("Win Rate [%]", "Win Rate"),
        ("Profit Factor", "Profit Factor"),
    ]

    def format_comparison_value(metric, value):
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            return "—"
        if metric in {"Start", "End"}:
            return str(value).split(" ")[0]
        if "[%]" in metric:
            return f"{float(value):.2f}%"
        if "[$]" in metric:
            return f"${float(value):,.2f}"
        if metric == "# Trades":
            return f"{int(value):,}"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.2f}"
        return str(value)

    comparison = {}
    for strategy_name, stats in results.items():
        comparison[strategy_name] = {
            display_name: format_comparison_value(metric, stats.get(metric))
            for metric, display_name in metrics
        }

    comparison = pd.DataFrame(comparison)
    comparison.index.name = "Metric"
    st.markdown(f"### {title}")
    st.dataframe(comparison, width="stretch")


def display_summary_highlights(results, title):
    highlight_metrics = [
        ("Return [%]", "Best Return", True, lambda value: f"{value:.2f}%"),
        ("Sharpe Ratio", "Best Sharpe", True, lambda value: f"{value:.2f}"),
        (
            "Max. Drawdown [%]",
            "Lowest Drawdown",
            True,
            lambda value: f"{value:.2f}%",
        ),
    ]

    st.markdown(f"### {title} Highlights")
    columns = st.columns(3)
    for column, (metric, label, maximize, formatter) in zip(columns, highlight_metrics):
        candidates = []
        for strategy_name, stats in results.items():
            value = stats.get(metric)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                candidates.append((strategy_name, numeric_value))

        if not candidates:
            column.metric(label, "—")
            column.caption("No data")
            continue

        strategy_name, value = (
            max(candidates, key=lambda candidate: candidate[1])
            if maximize
            else min(candidates, key=lambda candidate: candidate[1])
        )
        column.metric(label, formatter(value))
        column.caption(strategy_name)


if backtest_view == "Daily":
    display_strategy_legend()
    display_backtest_stats(bt_daily_atr_stats, "Short-Term Trend + ATR Stop")
    display_trade_log(bt_daily_atr_stats, "Short-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_daily_atr_stats, "Short-Term Trend + ATR Stop")
    bt_daily_atr.plot(filename="EmaCrossATRDaily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Short-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCrossATRDaily.html")

    display_backtest_stats(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop")
    display_trade_log(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop")
    bt_daily_30_60.plot(filename="EmaCross30_60ATRDaily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Long-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCross30_60ATRDaily.html")

    display_backtest_stats(bt_daily_stats, "Short-Term Trend + EMA Exit")
    display_trade_log(bt_daily_stats, "Short-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_daily_stats, "Short-Term Trend + EMA Exit")
    bt_daily.plot(filename="EmaCrossLongOnly.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Short-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly.html")

    display_backtest_stats(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit")
    display_trade_log(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit")
    bt_daily_long_30_60.plot(filename="EmaCrossLongOnly30_60Daily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Long-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly30_60Daily.html")

    daily_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_daily_atr_stats,
        "Long-Term Trend + ATR Stop": bt_daily_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_daily_stats,
        "Long-Term Trend + EMA Exit": bt_daily_long_30_60_stats,
    }
    display_strategy_comparison(daily_strategy_results, "Daily Strategy Comparison")

#####################################################################################

bt_weekly = FractionalBacktest(weekly1, EmaCrossLongOnly,cash=1000, commission=0.002,trade_on_close=True)
bt_weekly_stats = bt_weekly.run()
print("Weekly backtest results:")
print(bt_weekly_stats.to_string())

bt_weekly_atr = FractionalBacktest(weekly1, EmaCross,cash=1000, commission=0.002,trade_on_close=True)
bt_weekly_atr_stats = bt_weekly_atr.run()
print("Weekly EMA + ATR backtest results:")
print(bt_weekly_atr_stats.to_string())

bt_weekly_30_60 = FractionalBacktest(
    weekly1,
    EmaCross30_60,
    cash=1000,
    commission=0.002,
    trade_on_close=True,
)
bt_weekly_30_60_stats = bt_weekly_30_60.run()
print("Weekly EMA 30/60 + ATR backtest results:")
print(bt_weekly_30_60_stats.to_string())

bt_weekly_long_30_60 = FractionalBacktest(
    weekly1,
    EmaCrossLongOnly30_60,
    cash=1000,
    commission=0.002,
    trade_on_close=True,
)
bt_weekly_long_30_60_stats = bt_weekly_long_30_60.run()
print("Weekly EMA 30/60 Exit results:")
print(bt_weekly_long_30_60_stats.to_string())

if backtest_view == "Weekly":
    display_strategy_legend()
    display_backtest_stats(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop")
    display_trade_log(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop")
    bt_weekly_atr.plot(filename="EmaCrossATRWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Short-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCrossATRWeekly.html")

    display_backtest_stats(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop")
    display_trade_log(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop")
    bt_weekly_30_60.plot(filename="EmaCross30_60ATRWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Long-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCross30_60ATRWeekly.html")

    display_backtest_stats(bt_weekly_stats, "Short-Term Trend + EMA Exit")
    display_trade_log(bt_weekly_stats, "Short-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_weekly_stats, "Short-Term Trend + EMA Exit")
    bt_weekly.plot(filename="EmaCrossLongOnlyWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Short-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnlyWeekly.html")

    display_backtest_stats(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit")
    display_trade_log(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit")
    bt_weekly_long_30_60.plot(filename="EmaCrossLongOnly30_60Weekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Long-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly30_60Weekly.html")

    weekly_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_weekly_atr_stats,
        "Long-Term Trend + ATR Stop": bt_weekly_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_weekly_stats,
        "Long-Term Trend + EMA Exit": bt_weekly_long_30_60_stats,
    }
    display_strategy_comparison(weekly_strategy_results, "Weekly Strategy Comparison")

if backtest_view == "Summary":
    st.markdown("## Daily & Weekly Strategy Dashboard")
    st.caption("Compare the four strategies across both timeframes.")

    daily_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_daily_atr_stats,
        "Long-Term Trend + ATR Stop": bt_daily_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_daily_stats,
        "Long-Term Trend + EMA Exit": bt_daily_long_30_60_stats,
    }
    weekly_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_weekly_atr_stats,
        "Long-Term Trend + ATR Stop": bt_weekly_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_weekly_stats,
        "Long-Term Trend + EMA Exit": bt_weekly_long_30_60_stats,
    }
    display_summary_highlights(daily_strategy_results, "Daily")
    display_summary_highlights(weekly_strategy_results, "Weekly")
    display_strategy_comparison(daily_strategy_results, "Daily Strategy Comparison")
    display_strategy_comparison(weekly_strategy_results, "Weekly Strategy Comparison")