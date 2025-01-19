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
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

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

# [Previous functions remain the same: fetch_stock_data, calculate_technical_indicators, 
# generate_technical_analysis, generate_fundamental_analysis]

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
