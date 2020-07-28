from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import pickle as pi
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras import backend as K



def home(request):
    return render(request,'neuHome.html')

def ann(request):
    return render(request,'ann.html')


def cnn(request):
    return render(request,'cnn.html')

def rnn(request):
    return render(request,'rnn.html')



def annOut(request):
    #Impoting dataset
    dataset = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Neural_net/Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    #Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')
    X = np.array(transformer.fit_transform(X), dtype=np.float)
    X = X[:, 1:]

    #Splitting the dataset into training set anad test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

    #Feature Scaling (can be needed)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Neural_net/static/ann_model2','rb') as f:
        mp = pi.load(f)
    #Predicting the Test results
    y_pred = mp.predict(X_test)
    y_pred = (y_pred>0.5)
    print("**************")
    print(y_pred.shape)
    np.squeeze(y_pred,axis=(1,))
    print(y_pred.shape)
        #making the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return HttpResponse(y_pred)


def cnnOut(request):
    if request.method == 'POST' and request.FILES:
        #Before prediction
        K.clear_session()
        print("****************")
        # print(request.FILES['image'])
        img_width, img_height = 64, 64
        # img = request.FILES.get('fd'.get('img'), False)
        img = request.FILES.get('img')

        # img = request.POST.data['fd'] if request.POST.get('fd', False) else False
        print(img)
        with open('/home/grant/ADMIN/Project/internship19/ml_online-project/Neural_net/static/cnn_model2','rb') as fi:
            mp = pi.load(fi)
        mp.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

        img = image.load_img(img, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        ans = mp.predict(x)
        print(ans)
        if ans == [[1.]]:
            print("Dog")
            val = "It might be a dog"
        else:
            print("Cat")
            val = "It might be a cat"

        #After prediction
        K.clear_session()
    return HttpResponse(val)

def showCSV(request):
    if request.method == 'POST':
        import pandas as pd
        df = pd.read_csv('/home/grant/ADMIN/Project/internship19/ml_online-project/Neural_net/Churn_Modelling.csv')
        print(df)
        print("")
        print(df.to_html())
        return HttpResponse(df.to_html())
    else:
        return HttpResponse('unsuccessful')
