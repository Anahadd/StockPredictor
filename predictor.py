import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.Ticker(ticker)
    return stock_data.history(period='1d', start=start_date, end=end_date)

def preprocess_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data['Close'].iloc[i:i + window_size].values)
        y.append(data['Close'].iloc[i + window_size])
    return np.array(X), np.array(y)

ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'

data = get_stock_data(ticker, start_date, end_date)
X, y = preprocess_data(data, window_size=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')

plt.plot(y_test, label="True prices")
plt.plot(predictions, label="Predicted prices")
plt.legend()
plt.show()
