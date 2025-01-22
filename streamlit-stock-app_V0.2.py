import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="Stock Analysis AI", layout="wide")
st.title("Stock Analysis AI Assistant 📈")

# Sidebar for user input
st.sidebar.header("Analysis Parameters")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
)
analysis_type = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Technical Analysis", "Fundamental Analysis", "Sentiment Analysis", "Price Forecast"],
    default=["Technical Analysis"]
)

if "Price Forecast" in analysis_type:
    forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 7)

def fetch_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df, stock
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def calculate_technical_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def prepare_forecast_data(df, lookback=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    return X_train, y_train, X, scaler

def create_lstm_model(lookback):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_price_forecast(df, forecast_days):
    lookback = 60
    X_train, y_train, X, scaler = prepare_forecast_data(df, lookback)
    model = create_lstm_model(lookback)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    last_sequence = X[-1]
    future_predictions = []
    
    for _ in range(forecast_days):
        next_pred = model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
        future_predictions.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions))
    future_dates = [df.index[-1] + timedelta(days=x+1) for x in range(forecast_days)]
    
    return future_dates, future_predictions

def generate_technical_analysis(df):
    last_close = df['Close'].iloc[-1]
    ma20, ma50 = df['MA20'].iloc[-1], df['MA50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd, signal = df['MACD'].iloc[-1], df['Signal_Line'].iloc[-1]
    
    insights = []
    insights.append("📈 Upward trend: 20-day MA above 50-day MA" if ma20 > ma50 else "📉 Downward trend: 20-day MA below 50-day MA")
    
    if rsi > 70: insights.append("⚠️ Overbought: RSI above 70")
    elif rsi < 30: insights.append("🔔 Oversold: RSI below 30")
    else: insights.append("✅ Neutral RSI levels")
    
    insights.append("🚀 Bullish MACD crossover" if macd > signal else "🔻 Bearish MACD crossover")
    
    return insights

def generate_fundamental_analysis(stock):
    try:
        info = stock.info
        insights = []
        
        if 'forwardPE' in info:
            insights.append(f"📊 Forward P/E Ratio: {info['forwardPE']:.2f}")
        if 'marketCap' in info:
            insights.append(f"💰 Market Cap: ${info['marketCap']/1e9:.2f}B")
        if 'dividendYield' in info and info['dividendYield']:
            insights.append(f"💵 Dividend Yield: {info['dividendYield']*100:.2f}%")
        
        return insights
    except:
        return ["Unable to fetch fundamental data"]

def plot_stock_data(df, forecast_dates=None, forecast_values=None):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'], name='20-day MA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'], name='50-day MA',
        line=dict(color='blue')
    ))
    
    if forecast_dates and forecast_values:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values.flatten(),
            name='Price Forecast', line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price Analysis with Forecast',
        yaxis_title='Price', xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

# Main app logic
if stock_symbol:
    df, stock = fetch_stock_data(stock_symbol, time_period)
    
    if df is not None:
        df = calculate_technical_indicators(df)
        
        forecast_dates = None
        forecast_values = None
        if "Price Forecast" in analysis_type:
            with st.spinner("Generating price forecast..."):
                forecast_dates, forecast_values = generate_price_forecast(df, forecast_days)
        
        st.plotly_chart(plot_stock_data(df, forecast_dates, forecast_values), 
                       use_container_width=True)
        
        cols = st.columns(len(analysis_type))
        for i, analysis in enumerate(analysis_type):
            with cols[i]:
                st.subheader(analysis)
                
                if analysis == "Technical Analysis":
                    insights = generate_technical_analysis(df)
                elif analysis == "Fundamental Analysis":
                    insights = generate_fundamental_analysis(stock)
                elif analysis == "Price Forecast":
                    insights = [
                        f"🔮 Forecast for {forecast_days} days",
                        f"📈 Predicted closing price: ${forecast_values[-1][0]:.2f}",
                        f"📊 Average predicted price: ${np.mean(forecast_values):.2f}"
                    ]
                else:  # Sentiment Analysis
                    insights = ["Sentiment analysis coming soon!"]
                
                for insight in insights:
                    st.write(insight)
        
        st.subheader("Key Metrics")
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric(
                "Current Price",
                f"${df['Close'].iloc[-1]:.2f}",
                f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%"
            )
        with metrics_cols[1]:
            st.metric(
                "Volume",
                f"{df['Volume'].iloc[-1]:,.0f}",
                f"{((df['Volume'].iloc[-1] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2] * 100):.2f}%"
            )
        with metrics_cols[2]:
            st.metric(
                "RSI",
                f"{df['RSI'].iloc[-1]:.2f}",
                f"{(df['RSI'].iloc[-1] - df['RSI'].iloc[-2]):.2f}"
            )

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and Python")
