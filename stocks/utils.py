import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from .models import StockData  # Ensure you have the correct import path

# Function to retrieve stock data
def get_stock_data(symbol):
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')[:15]
    
    dates = []
    actual_prices = []
    predicted_prices = []

    for data in stock_data:
        dates.append(data.date)
        actual_prices.append(data.close_price)
        predicted_prices.append(data.predicted_price)
    
    return dates, actual_prices, predicted_prices

def calculate_metrics(actual_prices, predicted_prices):
    # Filter out None values from actual and predicted prices
    actual_prices = [price for price in actual_prices if price is not None]
    predicted_prices = [price for price in predicted_prices if price is not None]
    
    # If no valid prices are left, return default metrics
    if not actual_prices or not predicted_prices:
        return {
            'roi': 0,
            'total_return': 0,
            'max_drawdown': 0
        }

    # Step 1: Return on Investment (ROI)
    initial_price = actual_prices[0]
    final_price = actual_prices[-1]
    roi = (final_price - initial_price) / initial_price * 100
    
    # Step 2: Total Return (sum of differences between predicted and actual)
    total_return = sum(predicted_prices) - sum(actual_prices)
    
    # Step 3: Max Drawdown (largest price drop during the period)
    max_drawdown = min(predicted_prices) - max(actual_prices)

    return {
        'roi': roi,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }

# Function to create visualizations
def create_visualization(dates, actual_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='orange', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import base64
from reportlab.lib.utils import ImageReader

def generate_pdf_report(metrics, image_base64):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(100, 750, "Stock Performance Report")
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 700, f"Return on Investment (ROI): {metrics['roi']:.2f}%")
    pdf.drawString(100, 680, f"Total Return: {metrics['total_return']:.2f}")
    pdf.drawString(100, 660, f"Max Drawdown: {metrics['max_drawdown']:.2f}")
    
    # Decode the image base64 string and write it to a temporary file
    image_data = base64.b64decode(image_base64)
    image_buffer = BytesIO(image_data)

    # Draw the image on the PDF
    pdf.drawImage(ImageReader(image_buffer), 100, 400, width=400, height=200)
    
    pdf.showPage()
    pdf.save()
    
    buffer.seek(0)
    return buffer