from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home,name='clHome'),
    path('decision-tree',views.decTree,name='decTree'),
    path('kernel-SVM',views.k_svm,name='k_svm'),
    path('K-nearest_neighbours',views.knn,name='knn'),
    path('logistic_regression',views.logReg,name='logReg'),
    path('naive-bayes',views.naiveB,name='naiveB'),
    path('random-forest',views.ranF,name='ranF'),
    path('SVM',views.svm,name='svm'),

]
