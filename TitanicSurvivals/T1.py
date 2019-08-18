# -*- coding: utf-8 -*-
"""
Created on Wed May  9 07:09:41 2018

@author: Krris
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.info())
print(test.info())
y = train['Survived']

#a = train.iloc[:,1]
print(train['Cabin'].value_counts())
#Sex,Age,Fare,Pclass
train.loc[train.Age.isnull(),'Age'] = train['Age'].mean()
#train.loc[train.Cabin.isnull(),'Cabin'] = train['Cabin'].mean()
test.loc[test.Age.isnull(),'Age'] = train['Age'].mean()
#test.loc[test.Cabin.isnull(),'Cabin'] = train['Cabin'].mean()
test.loc[test.Fare.isnull(),'Fare'] = test['Fare'].mean()

train.loc[train['Sex']=='female','Sex'] = 0
train.loc[train['Sex']=='male','Sex'] = 1

test.loc[test['Sex']=='female','Sex'] = 0
test.loc[test['Sex']=='male','Sex'] = 1

X_train = train[['Sex','Age','Fare','Pclass','Parch']]

X_test = test[['Sex','Age','Fare','Pclass','Parch']]

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

result = pd.DataFrame(test['PassengerId'])
result['Survived']= y_pred

result.to_csv('t1.csv',index= False)



