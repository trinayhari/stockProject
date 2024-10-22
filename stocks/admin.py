from django.contrib import admin
from .models import StockData

class StockDataAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'close_price', 'predicted_price')
    list_filter = ('symbol',)
    search_fields = ('symbol',)

admin.site.register(StockData, StockDataAdmin)
