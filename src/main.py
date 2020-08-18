import numpy as np
import util
import network

def main():
    
    # loading features and labels
    X, y = util.load_data()
    
    # no of nodes in each layer first(784) and last(10) are fixed for this digit-recognization-problem.
    layers_size = [784, 100, 30, 10]

    # intializing
    nn = network.NeuralNetwork(layers_size)
    nn.isShuffle = True
    nn.isValidate = True
    
    # training the network
    nn.fit(X, y, ephocs=10)

    # some example predictions
    preds = [7, 3425, 14634, 27345, 38234]
    for i in preds:

        # if you want to see predicting digit, set 'show=' flag as True 
        print('[main]: actual: ',np.argmax(y[i])," | network predicted: ",nn.predict(X[i], show=False))

    # for predicting kaggle test_set and saving.
    saving_file_name = "numpy_nn_submission.csv"
    predict_for_kaggle_test_set(nn=nn, filename=saving_file_name)



def predict_for_kaggle_test_set(nn,filename):
    """
    this function is responsible for saving test predictions to given filename.
    
    Parameters:
        nn      :   object          (a trained neural network object)
        filename:   str             (filename for saving & identifying different test predictions)

    Returns:
        (no-returns)
    """
    
    kaggle_test_set = util.load_test_data()
    preds = []

    for i in kaggle_test_set:
        preds.append(nn.predict(i, show=False))

    util.save_predictions(preds, filename)



if __name__ == '__main__':
    main()
