from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class Predict_awarness(models.Model):


    RID= models.CharField(max_length=300)
    Education_Level= models.CharField(max_length=300)
    Institution_Type= models.CharField(max_length=300)
    Attack_Date= models.CharField(max_length=300)
    Sex= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Device= models.CharField(max_length=300)
    IT_Student= models.CharField(max_length=300)
    Location= models.CharField(max_length=300)
    Internet_Type= models.CharField(max_length=300)
    Network_Type= models.CharField(max_length=300)
    Url= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



