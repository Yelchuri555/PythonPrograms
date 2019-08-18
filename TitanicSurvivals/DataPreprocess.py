import pandas as pd
# from pandas.tools.plotting import scatter_matrix


train_set = pd.read_csv("train.csv")
test_set = pd.read_csv('test.csv')

# print train_set.head()
def predic_0(data):
    predictions = []
    for index, passenger in data.iterrows():
        predictions.append(0)

    return pd.Series(predictions)

#saving the target  value separetly
target = train_set['Survived']

passengerId = test_set['PassengerId']

train_new = train_set.drop(["Survived",'Ticket','Name'],axis=1)
print (train_new.head())
print (train_new.info())
print (train_new.describe())
train_new.loc[train_new['Fare'] > 10,'Fare'] = 1
train_new.loc[train_new['Fare']<10,'Fare'] = 0
#
import matplotlib.pyplot as plt
train_new.plot(kind ='scatter')
plt.show()



