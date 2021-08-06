from django.db import models
from datetime import datetime

# Create your models here.
class Records(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    first_name = models.CharField(max_length=50, null=True)
    last_name = models.CharField(max_length=50, null=True)
    residence = models.CharField(max_length=50, null=True)
    country = models.CharField(max_length=50, null=True)
    education = models.CharField(max_length=150, null=True)
    occupation = models.CharField(max_length=150, null=True)
    marital_status = models.CharField(max_length=50, null=True)
    file_image = models.ImageField(null=True,blank=True,upload_to="images/")
    bio = models.CharField(max_length=150, null=True)
    recorded_at = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.first_name
    class Meta:
        verbose_name_plural = "Records"
