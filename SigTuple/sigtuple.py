# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:18:50 2018

@author: Krrish
"""

import pandas as pd
import numpy as np

train = pd.read_csv("Donut Data Set - TrainingSet.csv - Donut Data Set - TrainingSet.csv.csv")
test = pd.read_csv("Donut Data Set - TestSet.csv - Donut Data Set - TestSet.csv.csv")
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
train['Location']=label.fit_transform(train['Location'])
test['Location'] = label.transform(test['Location'])

X = train.iloc[:,1:11]
y1 = train.iloc[:,11]
y2 = train.iloc[:,12]

corr_matrix = train.corr()

#applying backward elimination using stats model
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((749,1)).astype(int), values = X,axis = 1)

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y1,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,4,5,6,9,10]]
regressor_OLS = sm.OLS(endog = y1,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[1,2,4,5,6,9,10]]
regressor_OLS = sm.OLS(endog = y1,exog = X_opt).fit()
regressor_OLS.summary()

'''
Features Extracted From OLS Model
Donut Estimator 1
Donut Area of cross section	Donut 	
Donut area of central hole / Donut Area of circumscribed circle	
Donut  Estimator 2	
Donut  Estimator 3		
Donut volume Estimator 6	
Location
'''
features = ["Donut Estimator 1","Donut Area of cross section","Donut area of central hole / Donut Area of circumscribed circle",'Donut  Estimator 2','Donut  Estimator 3','Donut volume Estimator 6',"Location"]

#Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
#X = train[features]
X = train.iloc[:,[2,4,5,6,9,10]]
Y = y2
#Y = y1
validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'neg_mean_squared_error'

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('RF', RandomForestRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVM', SVR()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print (results)


X_test =test[features]

x_train = train.iloc[:,[1,2,4,5,6,9,10]]
y_train = y1
x_test = test.iloc[:,[1,2,4,5,6,9,10]]
y_test = test.iloc[:,11]
regressor = KNeighborsRegressor()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(mean_squared_error(y_test,y_pred)**0.5) #RMSE Value : 4.55382



#applying backward elimination using stats model
import statsmodels.formula.api as sm
X = train.iloc[:,1:11]
X = np.append(arr = np.ones((749,1)).astype(int), values = X,axis = 1)

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y2,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5,6,7,9,10]]
regressor_OLS = sm.OLS(endog = y2,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5,6,9,10]]
regressor_OLS = sm.OLS(endog = y2,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,3,4,5,6,9,10]]
regressor_OLS = sm.OLS(endog = y2,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,4,5,6,9,10]]
regressor_OLS = sm.OLS(endog = y2,exog = X_opt).fit()
regressor_OLS.summary()
    
    
'''
predictors for Donut Volume
Donut Area of cross section	Donut 	
Donut area of central hole / Donut Area of circumscribed circle	
Donut  Estimator 2	
Donut  Estimator 3		
Donut volume Estimator 6	
Location

'''
#X_test =test[features]

x_train2 = train.iloc[:,[2,4,5,6,9,10]]
y_train2 = y2
x_test2 = test.iloc[:,[2,4,5,6,9,10]]
y_test2 = test.iloc[:,11]

regressor2  = LinearRegression()
regressor2.fit(x_train2,y_train2)

y_pred2 = regressor2.predict(x_test2)

print(mean_squared_error(y_test2,y_pred2)**0.5) 








