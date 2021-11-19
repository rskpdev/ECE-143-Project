import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def normalize(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled


def df_encode(df, method):
    """
    this is to encode the input dataframe for the columns that are not numerical
    :param df: input dataframe
    :param method: can be one-hot-dummy,i.e [0,0,0,1] or simple label encoder, i.e: 0,1,2,3
    :param columns:list of strings, specified columns to be encoded
    :return: the encoded dataframe
    """
    if method == 'one-hot':
        return pd.get_dummies(df)

    elif method == 'label_encoder':
        le = LabelEncoder()
        df = df.apply(le.fit_transform)
        return df
