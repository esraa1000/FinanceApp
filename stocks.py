# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Constants
SEQUENCE_LENGTH = 30  # Define sequence length for LSTM

def load_stock_data(ticker, start="2020-01-01"):
    """
    Fetch historical stock data for a given ticker from Yahoo Finance.
    """
    df = yf.download(ticker, start=start)
    return df[['Close']]  # Keep only closing price

def preprocess_data(df, sequence_length=SEQUENCE_LENGTH):
    """
    Preprocess stock data for LSTM.
    """
    # Ensure the data is numeric
    df = df.select_dtypes(include=['number'])
    
    # Check for missing values
    if df.isnull().any().any():
        df = df.fillna(df.mean())  # Fill missing values with the mean
    
    # Initialize the scaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Prepare sequences for LSTM
    x, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        x.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length])
    
    x = np.array(x)
    y = np.array(y)
    
    return x, y, scaler

def build_lstm_model(input_shape):
    """
    Create an LSTM model for stock price prediction.
    """
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
    """
    Train the LSTM model on stock data with validation.
    """
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),  # Use validation data
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Print final training and validation loss
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"\n✅ Final Training Loss: {final_train_loss:.4f}")
    print(f"✅ Final Validation Loss: {final_val_loss:.4f}")

    return model, history

def predict_stock(model, scaler, df_test, sequence_length=SEQUENCE_LENGTH):
    """
    Predict stock prices using the trained LSTM model.
    """
    try:
        # Ensure the data is in the correct format
        if df_test.empty:
            raise ValueError("Test data is empty. Cannot make predictions.")
        
        # Preprocess the test data
        df_scaled = scaler.transform(df_test[['Close']])
        
        # Prepare the input data for the LSTM model
        x_test = []
        for i in range(len(df_scaled) - sequence_length):
            x_test.append(df_scaled[i:i + sequence_length])
        
        x_test = np.array(x_test)
        
        # Debugging: Print the shape of x_test
        print(f"Shape of x_test: {x_test.shape}")
        
        if x_test.shape[0] == 0:
            raise ValueError("x_test is empty. Check the sequence length or test data.")
        
        # Make predictions
        predictions = model.predict(x_test)
        
        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(predictions)
        
        # Return predictions and corresponding dates
        return df_test.index[sequence_length:], predictions.flatten()
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def get_predictions(symbol):
    """
    Fetch data, preprocess, and make predictions for a given stock symbol.
    """
    try:
        # Fetch data for the selected symbol
        df_stock = pd.DataFrame(data[symbol].Close).dropna()
        
        # Check if there's enough data for predictions
        if len(df_stock) < SEQUENCE_LENGTH:
            raise ValueError(f"Not enough data for {symbol}. Need at least {SEQUENCE_LENGTH} rows.")
        
        # Preprocess the data
        df_scaled = scaler.transform(df_stock[['Close']])
        
        # Prepare the input data for the LSTM model
        x_test = []
        for i in range(len(df_scaled) - SEQUENCE_LENGTH):
            x_test.append(df_scaled[i:i + SEQUENCE_LENGTH])
        
        x_test = np.array(x_test)
        
        # Debugging: Print the shape of x_test
        print(f"Shape of x_test for {symbol}: {x_test.shape}")
        
        if x_test.shape[0] == 0:
            raise ValueError(f"No valid sequences for {symbol}. Check the sequence length or data.")
        
        # Make predictions
        predictions = model.predict(x_test)
        
        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(predictions)
        
        # Get corresponding dates
        dates = df_stock.index[SEQUENCE_LENGTH:]
        
        return dates, predictions.flatten()
    
    except Exception as e:
        print(f"Error during prediction for {symbol}: {e}")
        return None, None

def plot_predictions(actual, predicted):
    """
    Plot actual vs. predicted stock prices.
    """
    plt.figure(figsize=(12,6))
    plt.plot(actual, label="Actual Prices", color='blue')
    plt.plot(predicted, label="Predicted Prices", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.show()

def plot_training_history(history):
    """
    Plot training loss and validation loss over epochs.
    """
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.show()

# MAIN EXECUTION FLOW
if __name__ == "__main__":
    # Load data
    ticker = "AAPL"  # Change this to any stock symbol
    df = load_stock_data(ticker)

    # Split data into training and validation sets
    split_ratio = 0.8
    train_size = int(len(df) * split_ratio)

    df_train = df[:train_size]
    df_val = df[train_size:]

    # Preprocess data
    x_train, y_train, scaler = preprocess_data(df_train)
    x_val, y_val, _ = preprocess_data(df_val)

    # Build & Train Model
    model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
    trained_model, history = train_model(model, x_train, y_train, x_val, y_val)

    # Plot Training History
    plot_training_history(history)

    # Make Predictions
    predictions = predict_stock(trained_model, scaler, df_val)

    # Plot Results
    plot_predictions(df_val['Close'].values[SEQUENCE_LENGTH:], predictions)