import pandas as pd

train_data = pd.read_csv('train.csv')

print (train_data.info())

test_data = pd.read_csv("test.csv")

print (test_data.info())
print ("Survived " + str((train_data['Survived']==1).sum()))
print ( (train_data['Sex']=='female').sum())

print( (train_data['Sex']=='male').sum())

test_data['Survival'] = 0
test_data.loc[test_data['Sex'] == 'female','Survival'] = 1
print ((test_data['Survival']==1).sum())
print ( (test_data['Sex']=='female').sum())

result = pd.DataFrame(test_data['PassengerId'])
result['Survived'] = test_data['Survival']

# result.to_csv('female.csv',index=False)

import matplotlib.pyplot as plt

plt.hist(train_data['Age'])
plt.show()