from django.urls import path,include
from . import views
from django.views.generic import RedirectView



urlpatterns = [
    path('',views.home,name = 'regHome'),
    path('polynomial-regression',views.poly,name='poly'),
    path(r'ajax/show_output',views.polyOut, name='showOut'),
    path(r'^favicon\.ico$',RedirectView.as_view(url='/static/images/favicon.ico')),

]
