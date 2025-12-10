import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. Carregar dados
# -------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# -------------------------------
# 2. Extrair título do nome
# -------------------------------
def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    return title

train['Title'] = train['Name'].apply(extract_title)
test['Title'] = test['Name'].apply(extract_title)

# -------------------------------
# 3. Mapear títulos raros
# -------------------------------
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
    'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
    'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
    'Jonkheer': 'Rare'
}
train['Title'] = train['Title'].map(title_map).fillna(train['Title'])
test['Title'] = test['Title'].map(title_map).fillna(test['Title'])

# -------------------------------
# 4. Preencher valores nulos
# -------------------------------
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# -------------------------------
# 5. Cabin como binária
# -------------------------------
train['Cabin'] = train['Cabin'].notnull().astype(int)
test['Cabin'] = test['Cabin'].notnull().astype(int)

# -------------------------------
# 6. Codificação de variáveis categóricas
# -------------------------------
encoder = LabelEncoder()
train['Sex'] = encoder.fit_transform(train['Sex'])
test['Sex'] = encoder.transform(test['Sex'])

train = pd.get_dummies(train, columns=['Embarked', 'Title'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked', 'Title'], drop_first=True)

# -------------------------------
# 7. Alinhar colunas entre train e test
# -------------------------------
train, test = train.align(test, join='left', axis=1, fill_value=0)

# -------------------------------
# 8. Normalização de variáveis numéricas
# -------------------------------
scaler = StandardScaler()
num_cols = ['Age', 'Fare']

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# -------------------------------
# 9. Separar features e target
# -------------------------------
y = train['Survived']
X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1, errors="ignore")
X_test = test.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1, errors="ignore")

# -------------------------------
# 10. Dividir treino/validação
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 11. Treinar Random Forest
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------------------------------
# 12. Avaliar no conjunto de validação
# -------------------------------
y_pred = rf_model.predict(X_val)

print("Acurácia:", accuracy_score(y_val, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_val, y_pred))
print("Relatório de classificação:\n", classification_report(y_val, y_pred))

# -------------------------------
# 13. Prever no conjunto de teste
# -------------------------------
test_predictions = rf_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)