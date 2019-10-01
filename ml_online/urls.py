
from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home,name='Home'),
    path('about/', views.about, name = 'About'),
    path('classification/', include('Classification.urls'), name='Classification'),
    path('clustering/', include('Clustering.urls')),
    path('regression/', include('Regression.urls')),
    path('neural-net/', include('Neural_net.urls')),
]
