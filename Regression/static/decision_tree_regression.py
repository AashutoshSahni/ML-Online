#Decisioon tree Regression
#Simple Linear Regression
#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impoting dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)'''

#Feature Scaling
'''#Feature Scaling (can be needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Simple Linear  Regression on to the trtainig set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

import pickle as pi
with open('dec_tree_model','wb') as f:
    pi.dump(regressor,f)


with open('dec_tree_model','rb') as fi:
    mp = pi.load(fi) 
    
y_mp = mp.predict([[6.5]])

#Predicting the test results
y_pred = regressor.predict(np.array(6.5).reshape(1,-1)) #reults the mean in the intervals

#Visualising the Trainig set results
#Visualising the Polynomial Reression Results(for higher smoother curves)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') #not to se X_poly
plt.title('Truth or Bluff (Decision Tree Reression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()   
