import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


class MultiColumnEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def encode(self, X):
        """
        Encode the specified columns of X  in dataframe using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        encoded = X.copy()
        if self.columns is not None:
            for col in self.columns:
                encoded[col] = LabelEncoder().fit_transform(encoded[col])
        else:
            for key, col in encoded.iteritems():
                encoded[key] = LabelEncoder().fit_transform(col)
        return encoded

    def encoder(self, X):
        return self.encode(X)


def df_encode(df, method, columns=None):
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
        return MultiColumnEncoder(columns=columns).encoder(df)
