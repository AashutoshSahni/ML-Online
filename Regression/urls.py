from django.urls import path,include
from . import views
from django.views.generic import RedirectView



urlpatterns = [
    path('',views.home,name = 'regHome'),
    path('simple-linear-regression',views.linR,name='lin'),
    path('multiple-linear-regression',views.multiR,name='multi'),
    path('polynomial-regression',views.poly,name='poly'),
    path('support-vector-regression',views.svr,name='svr'),
    path('decision-tree-regression',views.dtr,name='DTR'),
    path('random-forest-regression',views.rfr,name='RFR'),
    path(r'ajax/show_linOut',views.linOut, name='showLinOut'),
    path(r'ajax/show_polyOut',views.polyOut, name='showPolyOut'),
    path(r'ajax/show_svrOut',views.svrOut, name='showSvrOut'),
    path(r'ajax/show_dtrOut',views.dtrOut, name='showDtrOut'),
    path(r'ajax/show_rfrOut',views.rfrOut, name='showRfrOut'),
    path(r'ajax/show_csv',views.showCSV,name='showCSV'),
    path(r'ajax/show_mcsv',views.showMCSV,name='show_MCSV'),
    path(r'ajax/show_multiOut',views.multiOut,name='multiOut'),
    path(r'^favicon\.ico$',RedirectView.as_view(url='/static/images/favicon.ico')),

]
