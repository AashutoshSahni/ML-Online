from django.shortcuts import render
from django.http import HttpResponse
import io
import base64
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pylab
#Hierarchical Clustering

#Importing the libraries
import numpy as np
import pandas as pd
# Create your views here.
def home(request):
    return render(request,'clu_home.html')

def hc(request):
    return render(request,'hierar.html')

def kmeans(request):
    return render(request,'kmeans.html')
# def dendo(request):
#
#
#
#     #Importing the mall dataset
#     dataset = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Clustering/Mall_Customers.csv')
#     X = dataset.iloc[:, [3,4]].values
#
#     # Using dendrogram to find optimal number of clusters
#     import scipy.cluster.hierarchy as sch
#     dendogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward to minimize variance
#     plt.title("Dendrogram")
#     plt.xlabel('Customers')
#     plt.ylabel('Euclidean  distance')
#
#     # response = HttpResponse(content_type="image/png")
#     # pylab.savefig(response, format="png")
#     sio = io.BytesIO()
#     plt.savefig(sio, format="png")
#     encoded_img = base64.b64encode(sio.getvalue())
#     # return HttpResponse('<img src="data:image/png;base64,%s" />' %encoded_img)
#     return HttpResponse(encoded_img)
