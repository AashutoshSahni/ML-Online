#Simple Linear Regression
#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impoting dataset
dataset = pd.read_csv('Salary_Data.csv') 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#Feature Scaling
'''#Feature Scaling (can be needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Simple Linear  Regression on to the trtainig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

import pickle as pi
with open('linreg_model','wb') as f:
    pi.dump(regressor,f)


with open('linreg_model','rb') as fi:
    mp = pi.load(fi)    

#Predicting the test results
y_pred = regressor.predict(X_test)
y_mp = mp.predict([[8]])

#Visualising the Trainig set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience (Trainig set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()