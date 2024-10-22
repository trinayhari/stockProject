from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.FloatField(null=True, blank=True)  # Allow null values
    high_price = models.FloatField(null=True, blank=True)  # Allow null values
    low_price = models.FloatField(null=True, blank=True)   # Allow null values
    close_price = models.FloatField(null=True, blank=True) # Prediction target
    volume = models.IntegerField(null=True, blank=True)    # Allow null values
    predicted_price = models.FloatField(null=True, blank=True)  # Ensure this field is added
    class Meta:
        unique_together = ('symbol', 'date')
