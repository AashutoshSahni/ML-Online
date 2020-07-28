from django.shortcuts import render,redirect
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from django.views.decorators.csrf import csrf_protect



def home(request):
    return render(request,'regHome.html')

def linR(request):
    return render(request,'linReg.html')

def multiR(request):
    return render(request,'multiReg.html')

def poly(request):
    return render(request,'polyreg.html')

def svr(request):
    return render(request,'svr.html')

def dtr(request):
    return render(request,'decTreeReg.html')

def rfr(request):
    return render(request,'ranFoReg.html')

@csrf_protect
def linOut(request):
    if request.method == 'POST':
        if request.POST['num']:
            num =  request.POST['num']
            try:
                val = float(num)
            except ValueError:
                try:
                    val = int(num)
                except ValueError:
                    return HttpResponse("Please enter a number in range [1-10]")

            if val <= 10 and val>=1:

                import pickle as pi
                with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/linreg_model','rb') as fi:
                    mp = pi.load(fi)
                    y_mp = mp.predict([[val]])
                    print(num)
                    print(y_mp)
                    #print(x)
                    return HttpResponse(y_mp)
            else:
                return HttpResponse("Please enter a number in range [1-10]")
        else:
            return HttpResponse("NO")
    else:
     return HttpResponse('unsuccessful')


@csrf_protect
def polyOut(request):
    if request.method == 'POST':
        if request.POST['num']:
            num =  request.POST['num']
            try:
                val = float(num)
            except ValueError:
                try:
                    val = int(num)
                except ValueError:
                    return HttpResponse("Please enter a number in range [1-10]")

            if val <= 10 and val>=1:
                poly_reg = PolynomialFeatures(degree=4)
                x = poly_reg.fit_transform([[val]])
                import pickle as pi
                with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/polyreg_model','rb') as fi:
                    mp = pi.load(fi)
                    y_mp = mp.predict(x)
                    print(num)
                    print(y_mp)
                    print(x)
                    return HttpResponse(y_mp)
            else:
                return HttpResponse("Please enter a number in range [1-10]")
        else:
            return HttpResponse("NO")
    else:
     return HttpResponse('unsuccessful')

@csrf_protect
def svrOut(request):
    if request.method == 'POST':
        if request.POST['num']:
            num =  request.POST['num']
            try:
                val = float(num)
            except ValueError:
                try:
                    val = int(num)
                except ValueError:
                    return HttpResponse("Please enter a number in range [1-10]")

            if val <= 10 and val>=1:
                from sklearn.preprocessing import StandardScaler
                import pandas as pd
                dataset = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/Position_Salaries.csv')
                X = dataset.iloc[:, 1:2].values
                y = dataset.iloc[:, 2].values
                sc_y = StandardScaler()
                sc_X = StandardScaler()
                X = sc_X.fit_transform(X)
                y = sc_y.fit_transform(y.reshape(-1,1))

                import pickle as pi
                with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/svr_model','rb') as fi:
                    mp = pi.load(fi)
                    y_mp = sc_y.inverse_transform(mp.predict(sc_X.transform(np.array([[val]]))))
                    print(num)
                    print(y_mp)
                    #print(x)
                    return HttpResponse(y_mp)
            else:
                return HttpResponse("Please enter a number in range [1-10]")
        else:
            return HttpResponse("NO")
    else:
     return HttpResponse('unsuccessful')

@csrf_protect
def dtrOut(request):
    if request.method == 'POST':
        if request.POST['num']:
            num =  request.POST['num']
            try:
                val = float(num)
            except ValueError:
                try:
                    val = int(num)
                except ValueError:
                    return HttpResponse("Please enter a number in range [1-10]")

            if val <= 10 and val>=1:

                import pickle as pi
                with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/dec_tree_model','rb') as fi:
                    mp = pi.load(fi)
                    y_mp = mp.predict([[val]])
                    print(num)
                    print(y_mp)
                    #print(x)
                    return HttpResponse(y_mp)
            else:
                return HttpResponse("Please enter a number in range [1-10]")
        else:
            return HttpResponse("NO")
    else:
     return HttpResponse('unsuccessful')

@csrf_protect
def rfrOut(request):
    if request.method == 'POST':
        if request.POST['num']:
            num =  request.POST['num']
            try:
                val = float(num)
            except ValueError:
                try:
                    val = int(num)
                except ValueError:
                    return HttpResponse("Please enter a number in range [1-10]")

            if val <= 10 and val>=1:

                import pickle as pi
                with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/ran_forest_model','rb') as fi:
                    mp = pi.load(fi)
                    y_mp = mp.predict([[val]])
                    print(num)
                    print(y_mp)
                    #print(x)
                    return HttpResponse(y_mp)
            else:
                return HttpResponse("Please enter a number in range [1-10]")
        else:
            return HttpResponse("NO")
    else:
     return HttpResponse('unsuccessful')

def multiOut(request):
    #Impoting dataset
    dataset = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/50_Startups.csv')
    #Data Preprocessing


    #Impoting dataset
    # dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    #Enciding the categorical data
    #Encoding the independent variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    labelencoder_X = LabelEncoder()
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
    X = np.array(transformer.fit_transform(X), dtype=np.float)

    #Avoiding the dummy variable trap
    X = X[:,1:]

    #Splitting the dataset into training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



    #Fitting Multiple Linear regression to the trainig set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    #Predicting the test set results
    y_pred = regressor.predict(X_test)
    print(y_pred)



    return HttpResponse(y_pred)



def showCSV(request):
    if request.method == 'POST':

        df = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/Position_Salaries.csv')
        # print(df)
        # print("")
        # print(df.to_html())
        return HttpResponse(df.to_html())
    else:
        return HttpResponse('unsuccessful')

def showMCSV(request):
    if request.method == 'POST':

        df = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Regression/50_Startups.csv')
        # print(df)
        # print("")
        # print(df.to_html())
        return HttpResponse(df.to_html())
    else:
        return HttpResponse('unsuccessful')
