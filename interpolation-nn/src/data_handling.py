import numpy as np
from sklearn.model_selection import train_test_split

def data_to_numpy(df):
    X = df.drop(columns=['posterior', 'log_posterior']).to_numpy().astype(np.float32)
    Y = df['log_posterior'].to_numpy().astype(np.float32)
    return (X, Y)

def split_data(X, Y, train_size, valid_size, random_state=42):
    """
    Randomly split dataset, based on these ratios:
        'train': train_size
        'valid': valid_size
        'test':  1 - (train_size + valid_size)
    and divide into batches of size batch_size
    Uses internally train_test_split from sklearn.model_selection with stratify=None
    
    :param X, Y: data points to be split in X and corresponding labels in Y
    :type  X, Y: numpy arrays

    :param train_size: If float should be between 0.0 and 1.0

    :param valid_size: default = None
        If float should be between 0.0 and 1.0. 
        If None, only 'train' and 'test' will be returned

    :param random_state: Pass an int for reproducible output across multiple function calls.
    :type  random_state: int

    :return: splitting in tupel of [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]
             OR if valid_size = None
             splitting in tupel of [X_train, X_test, Y_train, Y_test]

    .. note::
        >>> split_data(data, 0.8, 0.05)
        Passing train_frac=0.8, valid_frac=0.05 gives a 80% / 5% / 15% split for train / validation / test
    """
    
    X_train, X_remain, Y_train, Y_remain = train_test_split(X, Y, train_size=train_size, random_state=random_state)

    if valid_size == None:
        return (X_train, X_remain, Y_train, Y_remain)
    else:
        valid_frac = valid_size/(1 - train_size)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_remain, Y_remain, train_size=valid_frac, random_state=random_state)
        return (X_train, X_valid, X_test, Y_train, Y_valid, Y_test)