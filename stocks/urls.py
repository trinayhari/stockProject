from django.urls import path
from .views import FetchStockDataView, BacktestView,PredictStockView,ViewStockData,PerformanceReportView

urlpatterns = [
    path('stocks/data/', ViewStockData.as_view(), name='view_stock_data'),
    path('predict/', PredictStockView.as_view(), name='predict_stock'),
    path('fetch/', FetchStockDataView.as_view(), name='fetch_stock_data'),
    path('backtest/', BacktestView.as_view(), name='backtest'),
    path('performance-report/', PerformanceReportView.as_view(), name='performance-report')
]

