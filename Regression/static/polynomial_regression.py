#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impoting dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)'''

'''#Feature Scaling (can be needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 



#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

import pickle as pi
with open('polyreg_model','wb') as f:
    pi.dump(lin_reg2,f)


with open('polyreg_model','rb') as fi:
    mp = pi.load(fi) 
    
y_mp = mp.predict(X_poly)
y_poly_pred = lin_reg2.predict(X_poly)

#Visualising the Linear Reression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Reression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()    

#Visualising the Polynomial Reression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') #not to se X_poly
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()    