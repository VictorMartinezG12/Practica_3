from django.db import models


class DetectionEvent(models.Model):
    objeto = models.CharField(max_length=100)
    confianza = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.objeto} ({self.confianza:.2f}) - {self.timestamp:%Y-%m-%d %H:%M:%S}"
