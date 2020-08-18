import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path


# function for loading data from disk
def load_data():
    """
    this function is responsible for loading traing data from disk.
    and performs some basic opertaions like
        - one-hot encoding
        - feature scaling
        - reshaping data

    Parameters:
        (no-parameters)

    Returns:
        X   :   numpy array         (contains all features of training data)
        y   :   numpy array         (contains all targets of traing data)
    
    """
    
    path = "../data/train.csv"
    
    if(not Path(path).is_file()):
        print("[util]: train data not found at '",path,"'")
        #quit()

    print("[util]: Loading '",path,"'")
    train = pd.read_csv(path)

    y = np.array(pd.get_dummies(train['label']))

    X = train.drop(['label'], axis=1)
    X = np.array(X/255)

    X = X.reshape(X.shape + (1,))
    y = y.reshape(y.shape + (1,))
    del train

    return X, y


# sigmoid activation function with derivative
def sigmoid(x, derivative=False):
    if(derivative):
        return sigmoid(x) * (1 - sigmoid(x))
    
    return 1.0/(1.0 + np.exp(-x))


# relu activation function with derivative
def relu(x, derivative=False):
    if(derivative):
        return x > 0
    
    return np.maximum(x, 0)


# softmax activation function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


# function for viewing digit from numpy array
def view_digit(x, title):
    """
    function for viewing one sample

    Parameters:
        x    :   numpy array         (contains one sample of all features)
        title:   string              (a predicted digit)

    Returns:
        (no-returns)  
    """    
    plt.matshow(x.reshape(28,28))
    plt.suptitle(" predicted as "+title)
    plt.show()


# function for shuffling the features and labels 
def shuffle(X, y):
    """
    function for shuffleing both features and targets.

    Parameters:
        X   :   numpy array         (contains all features)
        y   :   numpy array         (contains all targets)

    Returns:
        (no-returns)  
    """

    n = np.random.randint(1, 100)
    np.random.seed(n)
    np.random.shuffle(X)
    np.random.seed(n)
    np.random.shuffle(y)


# custom function for loading kaggle test data
def load_test_data():
    """
    this function is responsible for loading test data from disk.

    Parameters:
        (no-parameters)

    Returns:
        kt   :   numpy array         (contains all features of test data) 
    """
    path = "../data/test.csv"
    
    if(not Path(path).is_file()):
        print("[util]: test data not found at '",path,"'")
        #quit()

    print("[util]: Loading test data from: '",path,"'")
    test = pd.read_csv(path)
    kt = np.array(test/255)
    kt = kt.reshape(kt.shape + (1,))
    del test

    return kt

# custom function for saving kaggle test data predictions
def save_predictions(preds, filename='new_submission.csv'):
    """
    this function is responsible for saving test predictions to given filename.
    
    Parameters:
        preds   :   numpy array     (all the predictions of test set)
        filename:   str             (filename for saving & identifying different test predictions)

    Returns:
        (no-returns)
    """
    path = "../data/sample_submission.csv"
    
    if(not Path(path).is_file()):
        print("[util]: sample_submission file not found at '",path,"',\n\t it is required to get submission format")

    submission = pd.read_csv(path)
    submission['Label'] = preds
    submission.to_csv("../data/"+filename, index=False)

    print("[util]: predictions saved to:","../data/"+filename)


