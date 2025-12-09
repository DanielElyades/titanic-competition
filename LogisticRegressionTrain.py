import pandas as pd
import matplotlib.pyplot as plt
from pyexpat import features
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Cofigurações para exibir tudo

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


# Preencher os valores nulos no Arquivo train.csv
train['Age'].fillna(train['Age'].median(), inplace = True)
train['Cabin'] = train['Cabin'].fillna('U').astype(str).str[0]
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
# Preencher os valores nulos no Arquivo test.csv
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['Cabin'] = test['Cabin'].fillna('U').astype(str).str[0]

# Transformar variáveis categóricas em números
train['Sex'] = train['Sex'].map({'male':0, 'female':1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})
embarked_map = {'S':0, 'C':1, 'Q':2}
train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)
cabin_map = {letter:idx for idx, letter in enumerate(train['Cabin'].unique())}
train['Cabin'] = train['Cabin'].map(cabin_map)
test['Cabin'] = test['Cabin'].map(cabin_map)

# Selecionar variáveis para o modelo
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin']
x = train[features]
y = train['Survived']
x_test = test[features]

# Treinar o modelo simples
model = LogisticRegression(max_iter=200)
model.fit(x, y)
predictions = model.predict(x_test)

# Criar arquivo de submissão
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})
submission.to_csv('submission.csv', index=False)

#print(train.groupby('Sex')['Survived'].mean())
#print(train.groupby('Pclass')['Survived'].mean())
#print(train.groupby('Embarked')['Survived'].mean())

#print(test.isnull().sum())

#print(train['Sex'].value_counts())
#print(train['Sex'].value_counts(normalize=True))
#print(train.groupby('SibSp')['Survived'].mean())
#print(train[train['Survived']==1])

#print(train.head())
#print(test.head())

#print(train[train['Survived']==0])
#print(train.describe())

#print(train.isnull().sum())

#train['Age'].hist()
#train['Fare'].hist()
#plt.show()