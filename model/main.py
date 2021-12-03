import pandas as pd
from encoder import df_encode, normalize
from split_dataset import split
from models import *
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('heart.csv')
df = df.drop(['FastingBS', 'RestingBP'], axis=1)
df = df_encode(df, 'label_encoder')

x_train, x_test, y_train, y_test = split(df, 'HeartDisease')
x_train, x_test = normalize(x_train, x_test)
"""
a demo for model prediction, run all models and plot
"""
print(plot(x_train, y_train, x_test, y_test))
