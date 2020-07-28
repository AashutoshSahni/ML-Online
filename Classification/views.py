from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request,'cl_home.html')

def decTree (request):
    return render(request,'decT.html')

def k_svm (request):
    return render(request,'k_svm.html')

def knn (request):
    return render(request,'knn.html')

def logReg (request):
    return render(request,'logReg.html')

def naiveB (request):
    return render(request,'naive_bayes.html')

def ranF (request):
    return render(request,'ranF.html')

def svm (request):
    return render(request,'svm.html')
