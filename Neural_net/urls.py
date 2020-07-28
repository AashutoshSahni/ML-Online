from django.urls import path,include
from . import views
from django.views.generic import RedirectView


urlpatterns = [
    path('',views.home,name = 'neuHome'),
    path('CNN',views.cnn,name='cnn'),
    path('ANN',views.ann,name='ann'),
    path('RNN',views.rnn,name='rnn'),
    path(r'ajax/show_ANNOut',views.annOut,name='annOut'),
    path(r'ajax/show_CNNOut',views.cnnOut,name='cnnOut'),
    path(r'ajax/show_csv',views.showCSV,name='showCSV'),
    path(r'^favicon\.ico$',RedirectView.as_view(url='/static/images/favicon.ico')),
]
