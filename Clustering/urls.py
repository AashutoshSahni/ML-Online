from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home,name='cluHome'),
    path('hierarchical-clustering',views.hc,name='hc'),
    path('K-Means_clustering',views.kmeans,name='kmeans'),


]
