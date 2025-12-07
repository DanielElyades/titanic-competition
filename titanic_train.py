import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')

# Cofigurações para exibir tudo

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

print(train_data.groupby('Sex')['Survived'].mean())
print(train_data.groupby('Pclass')['Survived'].mean())
print(train_data.groupby('Embarked')['Survived'].mean())

#print(train_data['Sex'].value_counts())
#print(train_data['Sex'].value_counts(normalize=True))
#print(train_data.groupby('SibSp')['Survived'].mean())
#print(train_data[train_data['Survived']==1])
print(train_data.head())
#print(train_data[train_data['Survived']==0])
#print(train_data.describe())
print(train_data.isnull().sum())
train_data['Age'].hist()
train_data['Fare'].hist()
plt.show()