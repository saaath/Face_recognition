# Generated by Django 3.2 on 2021-07-09 15:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webcam', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='records',
            name='bio',
            field=models.CharField(max_length=150, null=True),
        ),
        migrations.AlterField(
            model_name='records',
            name='first_name',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
