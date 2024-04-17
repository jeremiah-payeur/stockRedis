import numpy as np
import yfinance as yf
from hmmlearn import hmm
import pandas as pd
import matplotlib.pyplot as plt

# Load historical stock data
def load_data(ticker, period='10y'):
    data = yf.Ticker(ticker).history(period=period)
    return data

# Feature engineering: Calculate percentage changes and additional indicators
def extract_features(data):
    data['PercentageChange'] = (data['Close'] - data['Open']) / data['Open']
    data['High_Var'] = (data['High'] - data['Open']) / data['Open']
    data['Low_Var'] = (data['Open'] - data['Low']) / data['Open']

    # Add more feature engineering here as needed
    return data

# Train Hidden Markov Model using all historical data
def train_hmm(data, n_components=3, n_iter=100):
    model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)

    # Extract features
    features = extract_features(data)

    # Fit the HMM model
    model.fit([features['PercentageChange'].values, features['High_Var'], features['Low_Var']])


    return model

# Predict using trained HMM model
def predict(model, data):
    features = extract_features(data)
    predicted_states = model.predict([features['PercentageChange']])
    return predicted_states

# Visualize predicted states and actual stock prices
def visualize_states(data, predicted_states):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.scatter(data.index, data['Close'], c=predicted_states, cmap='viridis', label='Predicted States', marker='o')
    plt.title('Predicted States vs. Actual Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Analyze characteristics of each state
def analyze_states(data, predicted_states):
    data['Predicted_State'] = predicted_states
    state_stats = data.groupby('Predicted_State')['PercentageChange'].agg(['mean', 'std', 'count'])
    print("State Statistics:")
    print(state_stats)

# Main function
def main():
    # Load historical stock data
    ticker = "AAPL"
    data = load_data(ticker)

    # Train HMM using all historical data
    n_components = 3  # Number of hidden states
    n_iter = 100  # Number of iterations for HMM training
    model = train_hmm(data, n_components, n_iter)

    # Predict using trained HMM model
    predicted_states = predict(model, data)

    # Visualize predicted states and actual stock prices
    visualize_states(data, predicted_states)

    # Analyze characteristics of each state
    analyze_states(data, predicted_states)

if __name__ == "__main__":
    main()

