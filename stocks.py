# # -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout

# # Constants
# SEQUENCE_LENGTH = 30  # Define sequence length for LSTM

# def load_stock_data(ticker, start="2020-01-01"):
#     """
#     Fetch historical stock data for a given ticker from Yahoo Finance.
#     """
#     df = yf.download(ticker, start=start)
#     return df[['Close']]  # Keep only closing price

# def preprocess_data(df, sequence_length=SEQUENCE_LENGTH):
#     """
#     Preprocess stock data for LSTM.
#     """
#     # Ensure the data is numeric
#     df = df.select_dtypes(include=['number'])
    
#     # Check for missing values
#     if df.isnull().any().any():
#         df = df.fillna(df.mean())  # Fill missing values with the mean
    
#     # Initialize the scaler
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df)
    
#     # Prepare sequences for LSTM
#     x, y = [], []
#     for i in range(len(df_scaled) - sequence_length):
#         x.append(df_scaled[i:i + sequence_length])
#         y.append(df_scaled[i + sequence_length])
    
#     x = np.array(x)
#     y = np.array(y)
    
#     return x, y, scaler

# def build_lstm_model(input_shape):
#     """
#     Create an LSTM model for stock price prediction.
#     """
#     model = Sequential()
#     model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=60, activation='relu', return_sequences=True))
#     model.add(Dropout(0.3))
#     model.add(LSTM(units=80, activation='relu', return_sequences=True))
#     model.add(Dropout(0.4))
#     model.add(LSTM(units=120, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(units=1))

#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# import numpy as np

# def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
#     """
#     Train the LSTM model on stock data.
#     """
#     try:
#         # Reshape x_train and x_test for LSTM input
#         print(f"Shape of x_train before reshaping: {x_train.shape}")
#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#         x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#         print(f"Shape of x_train after reshaping: {x_train.shape}")

#         # Train the model
#         history = model.fit(
#             x_train, y_train,
#             validation_data=(x_test, y_test),
#             epochs=epochs,
#             batch_size=batch_size,
#             verbose=1
#         )

#         # Print final training and validation loss
#         final_train_loss = history.history['loss'][-1]
#         final_val_loss = history.history['val_loss'][-1]

#         print(f"\n✅ Final Training Loss: {final_train_loss:.4f}")
#         print(f"✅ Final Validation Loss: {final_val_loss:.4f}")

#         return model, history

#     except Exception as e:
#         print(f"Error during training: {e}")
#         raise

# def predict_stock(model, scaler, df_test, sequence_length=SEQUENCE_LENGTH):
#     """
#     Predict stock prices using the trained LSTM model.
#     """
#     try:
#         # Ensure the data is in the correct format
#         if df_test.empty:
#             raise ValueError("Test data is empty. Cannot make predictions.")
        
#         # Preprocess the test data
#         df_scaled = scaler.transform(df_test[['Close']])
        
#         # Prepare the input data for the LSTM model
#         x_test = []
#         for i in range(len(df_scaled) - sequence_length):
#             x_test.append(df_scaled[i:i + sequence_length])
        
#         x_test = np.array(x_test)
        
#         # Debugging: Print the shape of x_test
#         print(f"Shape of x_test: {x_test.shape}")
        
#         if x_test.shape[0] == 0:
#             raise ValueError("x_test is empty. Check the sequence length or test data.")
        
#         # Make predictions
#         predictions = model.predict(x_test)
        
#         # Inverse transform the predictions to original scale
#         predictions = scaler.inverse_transform(predictions)
        
#         # Return predictions and corresponding dates
#         return df_test.index[sequence_length:], predictions.flatten()
    
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         raise


# def plot_predictions(actual, predicted):
#     """
#     Plot actual vs. predicted stock prices.
#     """
#     plt.figure(figsize=(12,6))
#     plt.plot(actual, label="Actual Prices", color='blue')
#     plt.plot(predicted, label="Predicted Prices", color='red', linestyle='dashed')
#     plt.xlabel("Time")
#     plt.ylabel("Stock Price")
#     plt.title("Stock Price Prediction")
#     plt.legend()
#     plt.show()

# def plot_training_history(history):
#     """
#     Plot training loss and validation loss over epochs.
#     """
#     plt.figure(figsize=(10,5))
#     plt.plot(history.history['loss'], label='Training Loss', color='blue')
#     plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training vs. Validation Loss")
#     plt.legend()
#     plt.show()

# # MAIN EXECUTION FLOW
# if __name__ == "__main__":
#     # Load data
#     ticker = "AAPL"  # Change this to any stock symbol
#     df = load_stock_data(ticker)

#     # Split data into training and validation sets
#     split_ratio = 0.8
#     train_size = int(len(df) * split_ratio)

#     df_train = df[:train_size]
#     df_val = df[train_size:]

#     # Preprocess data
#     x_train, y_train, scaler = preprocess_data(df_train)
#     x_val, y_val, _ = preprocess_data(df_val)

#     # Build & Train Model
#     model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
#     trained_model, history = train_model(model, x_train, y_train, x_val, y_val)

#     # Plot Training History
#     plot_training_history(history)

#     # Make Predictions
#     predictions = predict_stock(trained_model, scaler, df_val)

#     # Plot Results
#     plot_predictions(df_val['Close'].values[SEQUENCE_LENGTH:], predictions)

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_stock_data(ticker, start, end):
    """
    Fetch stock data from Yahoo Finance.
    """
    stock = yf.download(ticker, start=start, end=end)
    return stock

def preprocess_data(stock_data, seq_length=60):
    """
    Preprocess stock data: Normalize, create sequences, and split into training/testing sets.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock_data[['Adj Close']])

    # Split data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Function to create sequences
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """
    Train the LSTM model.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

def predict_stock(model, X_test, scaler):
    """
    Predict stock prices using the trained model.
    """
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)  # Convert back to original scale
