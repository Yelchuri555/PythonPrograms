import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
train = pd.read_csv('train.csv')
# print train.shape
test = pd.read_csv('test.csv')
print (len(test))
test['Survived'] = 'N'
# print test.shape
comb = pd.concat([train, test])
# print comb.shape
comb.loc[comb['Embarked'].isnull(),'Embarked'] = "S"
combine = [comb]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady','Dr', 'Sir', 'Countess', 'Capt', 'Col', 'Don', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')


    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# print comb[['Title', 'Survived']]

d = Counter(comb['Title'])
# print d

e = comb.groupby('Title', as_index=False)['Age'].median()
# print e
comb['TN']=comb['Title']
print (comb['TN'].head())

comb.loc[comb['Title']=='Master','Title']=4
comb.loc[comb['Title']=='Miss','Title']=22
comb.loc[comb['Title']=='Mr','Title']=29
comb.loc[comb['Title']=='Mrs','Title']=35
comb.loc[comb['Title']=='Rare','Title']=47.5

comb.Age.fillna(comb.Title, inplace=True)
# print comb.info()
comb.Age.fillna(comb.Title, inplace=True)

comb.loc[ comb['Age'] <= 16, 'Age'] = 0
comb.loc[(comb['Age'] > 16) & (comb['Age'] <= 32), 'Age'] = 1
comb.loc[(comb['Age'] > 32) & (comb['Age'] <= 48), 'Age'] = 2
comb.loc[(comb['Age'] > 48) & (comb['Age'] <= 64), 'Age'] = 3
comb.loc[ comb['Age'] > 64, 'Age'] = 4
# print comb

comb.loc[ comb['Sex'] =='female', 'Sex'] = 1
comb.loc[ comb['Sex'] =='male', 'Sex'] = 0

comb.loc[ comb['TN'] =='Master', 'TN'] = 0
comb.loc[ comb['TN'] =='Miss', 'TN'] = 1
comb.loc[ comb['TN'] =='Mr', 'TN'] = 2
comb.loc[ comb['TN'] =='Mrs', 'TN'] = 3
comb.loc[ comb['TN'] =='Rare', 'TN'] = 4

comb.loc[ comb['Embarked'] =='S', 'Embarked'] = 0
comb.loc[ comb['Embarked'] =='C', 'Embarked'] = 1
comb.loc[ comb['Embarked'] =='Q', 'Embarked'] = 2

comb['Fare'].fillna(comb['Fare'].dropna().median(), inplace=True)

comb.loc[comb['Fare'] <= 7.91, 'Fare'] = 0
comb.loc[(comb['Fare'] > 7.91) & (comb['Fare'] <= 14.454), 'Fare'] = 1
comb.loc[(comb['Fare'] > 14.454) & (comb['Fare'] <= 31), 'Fare']   = 2
comb.loc[comb['Fare'] > 31, 'Fare'] = 3
# print comb

comb['Age*Class'] = comb.Age * comb.Pclass
comb['Fare'] = comb['Fare']
print (comb['Age*Class'].head())
comb['Age*Class'] = comb['Age*Class'].astype(int)
comb['Age'] = comb['Age'].astype(int)
# print comb

train_n = comb.iloc[0:891,]
# print train_n.shape


test_n = comb.iloc[891:,]
# print test_n.shape

train_n1 = pd.concat([train_n['Pclass'],train_n['Sex'],train_n['Age'],train_n['SibSp'],train_n['Parch'],train_n['Fare'],train_n['Embarked'],train_n['TN'],train_n['Age*Class'],train_n['Survived']],axis=1)
# print train_n
train_n1['PassengerId'] = train['PassengerId']
test_n1 = pd.concat([test_n['Pclass'],test_n['Sex'],test_n['Age'],test_n['SibSp'],test_n['Parch'],test_n['Fare'],test_n['Embarked'],test_n['TN'],test_n['Age*Class'],test_n['PassengerId']],axis=1)
train_n1.to_csv('new_train.csv',index=False)
test_n1['PassengerId'] = test['PassengerId']
test_n1.to_csv('new_test.csv',index=False)
# print train_n1,type(train_n1)

train1 = pd.read_csv('new_train.csv')
print ('*************')
print (train1.info())
print ('*************')
test1 = pd.read_csv('new_test.csv')
print (len(test1))
arraytest = test1.values
comb['Age*Class'] = comb.Age * comb.Pclass

array = train1.values
X = array[:,0:9]
Y = array[:,9]
Xtest = arraytest[:,0:9]
print (Xtest.shape)
# print (X, type(X))
# print (Y,type(Y))
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Spot Check Algorithms

models = []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print (results)
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()