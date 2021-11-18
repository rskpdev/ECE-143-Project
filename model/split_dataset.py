from sklearn.model_selection import train_test_split


def split(df, label, test_size=0.2, random_state=25):
    """
    split the dataframe to training dataset and testing dataset
    :param df: input dataframe
    :param label: the label to be classified
    :param test_size: percentage of data to be tested
    :param random_state: random_state
    :return: the training datasets and testing datasets
    """
    X = df.drop(columns=[label])
    y = df[label]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
