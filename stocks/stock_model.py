import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from .models import StockData
from datetime import datetime

def train_model(symbol='AAPL'):
    # Fetch historical data for the specified stock
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))

    if df.empty:
        return None  # No data available

    # Prepare the data for the model
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].map(datetime.toordinal)  # Convert dates to ordinal
    X = df['date'].values.reshape(-1, 1)  # Features (dates)
    y = df['close_price'].values  # Target (closing prices)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_prices(model, start_date, num_days=30):
    future_dates = [start_date + i for i in range(1, num_days + 1)]
    future_dates_reshaped = np.array(future_dates).reshape(-1, 1)
    predicted_prices = model.predict(future_dates_reshaped)
    return predicted_prices, future_dates
