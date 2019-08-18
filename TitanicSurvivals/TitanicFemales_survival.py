# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 07:09:20 2018

@author: Krris
"""

import pandas as pd


train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

print(train.info())

print(test.info())

test['Survived'] = test['Sex']

test.loc[test.Survived =='female','Survived'] = 1
test.loc[test.Survived == 'male','Survived'] = 0

result = pd.DataFrame(test[['PassengerId','Survived']])


result.to_csv('result24.csv',index=False)


    

