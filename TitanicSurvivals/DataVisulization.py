import pandas as pd

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv("test.csv")


# print test_set.describe()
# print test_set.info()

list_drop = ['Name','Ticket','Cabin']
train_new = train_set.drop(list_drop,axis=1)
test_new = test_set.drop(list_drop,axis=1)

target = train_set['Survived']
train_new = train_new.drop('Survived',axis=1)


#filling Missing values
import numpy as np
test_new.loc[test_new['Fare'].isnull() ,'Fare'] = test_new['Fare'].mean()
train_set['Age'] = train_set['Age'] - 18
train_set.loc[train_set['Age'] < 19,'Age'] = 0
# train_set['Age'] = train_set['Age'] - (min(train_set['Age']))
train_set['Age'] = train_set['Age']/62
train_set['Age'] = train_set['Age'] *10
print("min of Fare"+str(min(train_set['Age'])))
print("Max of Fare"+str(max(train_set['Age'])))
train_new.loc[train_new['Fare'] > 0,'Fare'] = np.floor(train_new['Fare']/10)
test_new.loc[test_new['Fare'] >0 ,'Fare'] = np.floor(test_new['Fare']/10)

train_new.loc[train_new['Age'].isnull(),'Age'] = 28
test_new.loc[test_new['Age'].isnull(),'Age'] = 28
train_new.loc[train_new['Age'] > 0,'Age'] = np.floor(train_new['Age']/10)
test_new.loc[test_new['Age'] >0 ,'Age'] = np.floor(test_new['Age']/10)

#preprocessing categorical values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label_Encoder = LabelEncoder()
X = train_new.iloc[:,1:].values
Y = test_new.iloc[:,1:].values
for i in range(7):
    X[:, i] = label_Encoder.fit_transform(X[:, i])
    Y[:, i] = label_Encoder.fit_transform(Y[:, i])


train = pd.DataFrame(X)
test = pd.DataFrame(Y)
train.columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
test.columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']


print (train_new.head())
print (test_new.head())
# train = train.drop('SibSp',axis=1)
# test = test.drop('SibSp',axis=1)

# test.to_csv('test_new.csv')

train1 = train.drop(['Fare','Parch'],axis=1)
test1 = test.drop(['Fare','Parch'],axis=1)

print (train1.head())
print (test1.head())

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


logistic_Regression = DecisionTreeClassifier()

logistic_Regression.fit(train1.iloc[:,1:4],target)

y_predict = logistic_Regression.predict(test1.iloc[:,1:4])

sub = pd.DataFrame(test_new['PassengerId'])
sub['Survived'] = y_predict
# print train.iloc[0:5,1] sex

sub.to_csv('sub.csv')
