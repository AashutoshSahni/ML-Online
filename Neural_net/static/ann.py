#Artificial Neural Network 

#Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tensorflow
#Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.11/get_started
#Installing Keras

#Part 1 Data Preprocessing
#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impoting dataset
dataset = pd.read_csv('Churn_Modelling.csv') 
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

#Part 2 - Making ANN

#Importing keras libraries and packages
import keras   
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential() 

#Adding the input layer and first hidden layer #releu = rectifier layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation='relu', input_dim = 11))#using av= in+out/2 = 11+1/2=6

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation='relu'))#using av= in+out/2 = 11+1/2=6

#Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform', activation='sigmoid'))#using av= in+out/2 = 11+1/2=6

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#Part 3 - Making the predictions and evaluating the model

#Fitting Classifier to the Training Set
#Create your classifier here
import pickle as pi
with open('ann_model2','wb') as f: 
    pi.dump(classifier,f)



#Predicting the Test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
