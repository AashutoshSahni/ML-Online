from django.shortcuts import render,redirect
from django.http import HttpResponse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from django.views.decorators.csrf import csrf_protect



def home(request):
    return render(request,'regHome.html')


def poly(request):
    return render(request,'polyreg.html')

@csrf_protect
def polyOut(request):
    if request.method == 'POST':
         print("hiiiiii")
         if request.POST['num']:
            num =  request.POST['num']
            poly_reg = PolynomialFeatures(degree=4)
            x = poly_reg.fit_transform([[6.5]])
            import pickle as pi
            with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/polyreg_model','rb') as fi:
             mp = pi.load(fi)
            y_mp = mp.predict(x)
            print(num)
            print(y_mp)
            print(x)
            return HttpResponse(y_mp)
         else:
            print("no num")
    else:
     return HttpResponse('unsuccessful')

def showCSV(request):
    if request.method == 'POST':
        import pandas as pd
        columns = ['position', 'level', 'salary',]
        df = pd.read_csv('Position_Salaries.csv', names=columns)
        print(df)
        return HttpResponse('success')
    else:
        return HttpResponse('unsuccessful')    
