import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Stock Analysis AI", layout="wide")
st.title("Stock Analysis AI Assistant 📈")

# Sidebar for user input
st.sidebar.header("Analysis Parameters")

# Stock symbol input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

# Time period selection
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
)

# Analysis type selection
analysis_type = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Technical Analysis", "Fundamental Analysis", "Sentiment Analysis", "Price Forecast"],
    default=["Technical Analysis"]
)

# Forecasting parameters (only show if Price Forecast is selected)
if "Price Forecast" in analysis_type:
    forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 7)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(symbol, period):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            st.error("No data found for the given symbol.")
            return None, None
        return df, stock
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Moving averages
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

def generate_technical_analysis(df):
    """Generate technical analysis insights"""
    last_close = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    
    insights = []
    
    # Trend analysis
    if ma20 > ma50:
        insights.append("📈 Upward trend: 20-day MA above 50-day MA")
    else:
        insights.append("📉 Downward trend: 20-day MA below 50-day MA")
    
    # RSI analysis
    if rsi > 70:
        insights.append("⚠️ Overbought: RSI above 70")
    elif rsi < 30:
        insights.append("🔔 Oversold: RSI below 30")
    else:
        insights.append("✅ Neutral RSI levels")
    
    # MACD analysis
    if macd > signal:
        insights.append("🚀 Bullish MACD crossover")
    else:
        insights.append("🔻 Bearish MACD crossover")
    
    return insights

def generate_fundamental_analysis(stock):
    """Generate fundamental analysis insights"""
    try:
        info = stock.info
        insights = []
        
        # P/E Ratio analysis
        if 'forwardPE' in info:
            pe_ratio = info['forwardPE']
            insights.append(f"📊 Forward P/E Ratio: {pe_ratio:.2f}")
        
        # Market Cap analysis
        if 'marketCap' in info:
            market_cap = info['marketCap'] / 1e9  # Convert to billions
            insights.append(f"💰 Market Cap: ${market_cap:.2f}B")
        
        # Dividend analysis
        if 'dividendYield' in info and info['dividendYield']:
            div_yield = info['dividendYield'] * 100
            insights.append(f"💵 Dividend Yield: {div_yield:.2f}%")
        
        return insights
    except:
        return ["Unable to fetch fundamental data"]

def prepare_forecast_data(df, lookback=60):
    """Prepare data for LSTM model"""
    # Use closing price for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Prepare sequences for LSTM
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    return X_train, y_train, X, scaler

def create_lstm_model(lookback):
    """Create and compile LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

@st.cache_resource  # Cache the model to avoid retraining
def generate_price_forecast(df, forecast_days):
    """Generate price forecasts using LSTM"""
    lookback = 60
    
    # Prepare data
    X_train, y_train, X, scaler = prepare_forecast_data(df, lookback)
    
    # Create and train model
    model = create_lstm_model(lookback)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Generate predictions
    last_sequence = X[-1]
    future_predictions = []
    
    for _ in range(forecast_days):
        # Predict next value
        next_pred = model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
        future_predictions.append(next_pred[0])
        
        # Update sequence for next prediction
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    # Inverse transform predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions))
    
    # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=x+1) for x in range(forecast_days)]
    
    return future_dates, future_predictions

def plot_stock_data_with_forecast(df, forecast_dates=None, forecast_values=None):
    """Create interactive stock price plot with optional forecast"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA20'],
        name='20-day MA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        name='50-day MA',
        line=dict(color='blue')
    ))
    
    # Add forecast if available
    if forecast_dates is not None and forecast_values is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values.flatten(),
            name='Price Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price Analysis with Forecast',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

# Main app logic
if stock_symbol:
    df, stock = fetch_stock_data(stock_symbol, time_period)
    
    if df is not None:
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Generate and display forecast if selected
        forecast_dates = None
        forecast_values = None
        if "Price Forecast" in analysis_type:
            with st.spinner("Generating price forecast..."):
                forecast_dates, forecast_values = generate_price_forecast(df, forecast_days)
        
        # Display stock price chart with forecast
        st.plotly_chart(plot_stock_data_with_forecast(df, forecast_dates, forecast_values), 
                       use_container_width=True)
        
        # Display analysis based on user selection
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
                        f"📈 Predicted closing price (end of forecast): ${forecast_values[-1][0]:.2f}",
                        f"📊 Average predicted price: ${np.mean(forecast_values):.2f}"
                    ]
                else:  # Sentiment Analysis
                    insights = ["Sentiment analysis coming soon!"]
                
                for insight in insights:
                    st.write(insight)
        
        # Additional metrics
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and Python")