import pandas as pd
from encoder import df_encode, normalize
from split_dataset import split
from models import *
df = pd.read_csv('heart.csv')
df = df_encode(df, 'label_encoder')
x_train, x_test, y_train, y_test = split(df, 'HeartDisease')
x_train, x_test = normalize(x_train, x_test)
"""
a demo for model prediction, logistic regression model
"""
print(Logistic_Regression_Model(x_train, y_train, x_test, y_test))
