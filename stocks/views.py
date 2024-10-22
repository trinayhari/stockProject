from matplotlib import pyplot as plt
import requests
import numpy as np
from django.http import JsonResponse
from django.views import View
from .models import StockData
from django.shortcuts import render
import numpy as np
from datetime import timedelta, date
from .stock_model import train_model, predict_prices
from sklearn.linear_model import LinearRegression  # Add this line

class FetchStockDataView(View):
    def get(self, request):
        symbol = "AAPL"
        api_key = "1EYCRIBEPH84AHGX"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if "Time Series (Daily)" in data:
                for date, metrics in data["Time Series (Daily)"].items():
                    StockData.objects.create(
                        date=date,
                        open_price=float(metrics["1. open"]),
                        close_price=float(metrics["4. close"]),
                        high_price=float(metrics["2. high"]),
                        low_price=float(metrics["3. low"]),
                        volume=int(metrics["5. volume"]),
                    )
                return JsonResponse({"status": "success"})
            else:
                return JsonResponse({"status": "error", "message": "Invalid data received from API"}, status=400)
        
        except requests.exceptions.HTTPError as http_err:
            return JsonResponse({"status": "error", "message": str(http_err)}, status=500)
        except requests.exceptions.ConnectionError:
            return JsonResponse({"status": "error", "message": "Connection error occurred"}, status=500)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

class BacktestView(View):
    def get(self, request):
        return render(request, 'stocks/backtest.html', {
            'initial_investment': 1000,
            'symbol': 'AAPL',
        })

    def post(self, request):
        initial_investment = float(request.POST.get('initial_investment', 1000))
        symbol = request.POST.get('symbol', 'AAPL')
        
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        
        # Filter out None values in closing prices
        closing_prices = [data.close_price for data in stock_data if data.close_price is not None]

        if len(closing_prices) < 200:
            return render(request, 'stocks/backtest.html', {
                'result': {'status': 'error', 'message': 'Not enough data for backtesting.'},
                'initial_investment': initial_investment,
                'symbol': symbol,
            })

        # Calculate moving averages
        moving_average_50 = np.convolve(closing_prices, np.ones(50) / 50, mode='valid')
        moving_average_200 = np.convolve(closing_prices, np.ones(200) / 200, mode='valid')

        if len(moving_average_50) < 1 or len(moving_average_200) < 1:
            return render(request, 'stocks/backtest.html', {
                'result': {'status': 'error', 'message': 'Moving averages could not be calculated.'},
                'initial_investment': initial_investment,
                'symbol': symbol,
            })
        
        cash = initial_investment
        shares = 0
        trades = 0
        portfolio_values = []

        for i in range(len(moving_average_200), len(closing_prices)):
            portfolio_value = cash + shares * closing_prices[i]
            portfolio_values.append(portfolio_value)

            if i - 50 >= 0 and closing_prices[i] < moving_average_50[i - 50]:
                shares_to_buy = cash // closing_prices[i]
                cash -= shares_to_buy * closing_prices[i]
                shares += shares_to_buy
                trades += 1
            
            if i - 200 >= 0 and closing_prices[i] > moving_average_200[i - 200]:
                cash += shares * closing_prices[i]
                shares = 0
                trades += 1

        portfolio_value = cash + shares * closing_prices[-1]
        portfolio_values.append(portfolio_value)

        total_return = portfolio_value - initial_investment
        max_drawdown = calculate_max_drawdown(portfolio_values)

        return render(request, 'stocks/backtest.html', {
            'result': {
                'status': 'success',
                'total_return': total_return,
                'trades': trades,
                'max_drawdown': max_drawdown,
            },
            'initial_investment': initial_investment,
            'symbol': symbol,
        })

def calculate_max_drawdown(prices):
    peak = prices[0]
    max_drawdown = 0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


from django.http import JsonResponse
from django.views import View
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
from .models import StockData  # Make sure to import your StockData model

from django.http import JsonResponse
from django.views import View
from django.utils import timezone
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import date, timedelta
from .models import StockData

class PredictStockView(View):
    def get(self, request, *args, **kwargs):
        symbol = 'AAPL'  # You can pass this symbol via request if needed
        
        # Fetch stock data from the database
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        
        if not stock_data.exists():
            return JsonResponse({"status": "error", "message": "No stock data available for the given symbol."})

        # Prepare data for linear regression
        dates = np.array([i.toordinal() for i in stock_data.values_list('date', flat=True)])
        close_prices = np.array(stock_data.values_list('close_price', flat=True), dtype=float)  # Ensure it's float type

        # Check for NaN values in close_prices
        if np.isnan(close_prices).any():
            print("Found NaN values in close prices. Filtering out NaNs...")
            
            # Create a mask to filter out NaN values
            valid_indices = ~np.isnan(close_prices)
            dates = dates[valid_indices]
            close_prices = close_prices[valid_indices]

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(dates.reshape(-1, 1), close_prices)

        # Get today's date and the last date in the dataset
        today = date.today()
        last_date = stock_data.last().date

        # Create a list to hold all dates to predict: 15 days before today + 30 days after last_date
        predictions = []

        # Generate predictions for 15 days before today
        for i in range(-15, 0):  # 15 days before today
            future_date = today + timedelta(days=i)  # Generate past dates
            future_date_ordinal = np.array([[future_date.toordinal()]])
            predicted_price = model.predict(future_date_ordinal)[0]

            # Only store predictions if the price is not None
            if predicted_price is not None:
                # Check if the entry already exists
                if not StockData.objects.filter(symbol=symbol, date=future_date).exists():
                    # Save predictions to the database
                    StockData.objects.create(
                        symbol=symbol,
                        date=future_date,
                        predicted_price=predicted_price
                    )
                else:
                    # Optionally, you can update the existing record
                    StockData.objects.filter(symbol=symbol, date=future_date).update(predicted_price=predicted_price)

                # Add to response for display
                predictions.append({
                    "date": future_date.strftime('%Y-%m-%d'),  # Convert datetime to string
                    "predicted_price": predicted_price
                })

        # Generate predictions for today and 30 days after the last date
        for i in range(0, 31):  # Today and 30 days after today
            future_date = today + timedelta(days=i)
            future_date_ordinal = np.array([[future_date.toordinal()]])
            predicted_price = model.predict(future_date_ordinal)[0]

            # Check if we have actual price for today
            if i == 0 and not StockData.objects.filter(symbol=symbol, date=today).exists():
                # If there's no actual price for today, set predicted price to None (N/A)
                predicted_price = None

            # Only store predictions if the price is not None
            if predicted_price is not None:
                # Check if the entry already exists
                if not StockData.objects.filter(symbol=symbol, date=future_date).exists():
                    # Save predictions to the database
                    StockData.objects.create(
                        symbol=symbol,
                        date=future_date,
                        predicted_price=predicted_price
                    )
                else:
                    # Optionally, you can update the existing record
                    StockData.objects.filter(symbol=symbol, date=future_date).update(predicted_price=predicted_price)

                # Add to response for display
                predictions.append({
                    "date": future_date.strftime('%Y-%m-%d'),
                    "predicted_price": predicted_price
                })

        # Prepare data for rendering
        historical_data = {data.date.strftime('%Y-%m-%d'): data.close_price for data in stock_data}
        combined_data = {date: {'actual_price': historical_data.get(date, None), 'predicted_price': None} for date in [p['date'] for p in predictions]}

        # Update combined_data with predicted prices
        for predicted in predictions:
            combined_data[predicted['date']]['predicted_price'] = predicted['predicted_price']

        return JsonResponse({
            "symbol": symbol,
            "predictions": predictions
        })
    
from django.http import JsonResponse
from django.views import View
from .models import StockData

class ViewStockData(View):
    def get(self, request, *args, **kwargs):
        symbol = 'AAPL'  # This can be dynamic based on the request
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')

        data = []
        for record in stock_data:
            data.append({
                "symbol": record.symbol,
                "date": record.date.strftime('%Y-%m-%d'),
                "open_price": record.open_price,
                "high_price": record.high_price,
                "low_price": record.low_price,
                "close_price": record.close_price,
                "volume": record.volume,
                "predicted_price": record.predicted_price
            })

        return JsonResponse({
            "status": "success",
            "data": data,
        })

from django.views import View
from django.http import HttpResponse
from .utils import get_stock_data, calculate_metrics, create_visualization, generate_pdf_report

class PerformanceReportView(View):
    def get(self, request):
        symbol = 'AAPL'  # Set your stock symbol
        dates, actual_prices, predicted_prices = get_stock_data(symbol)

        # Calculate metrics
        metrics = calculate_metrics(actual_prices, predicted_prices)

        # Create visualization
        image_base64 = create_visualization(dates, actual_prices, predicted_prices)

        # Generate PDF report
        pdf_buffer = generate_pdf_report(metrics, image_base64)

        # Return PDF response
        return HttpResponse(pdf_buffer, content_type='application/pdf')
