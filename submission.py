import pandas as pd

from LogisticRegressionTrain import submission

submission = pd.read_csv('submission.csv')

print(submission.head())