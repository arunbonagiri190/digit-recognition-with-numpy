import numpy as np
import util

class NeuralNetwork:
    
    # hyperameters
    
    learning_rate = 0.0001
    l0, l1, l2, l3 =  0, 0, 0, 0

    isShuffle = False       # shuffle flag:    for shuffling data while training the network
    isValidate = False      # validation flag: for viewing validation results while training the network
    X_val = np.array([])   
    y_val = np.array([])


    def __init__(self, layers_size):
        
        """
            this is network cunstruct method, here we intializing the weights and biases

            Parameters:
                layers_size : list      (list integers, each integer coresponds to no.of nodes in each layer)

            Returns:
                (no-returns)
        """
        
        self.l0 = layers_size[0]
        self.l1 = layers_size[1]
        self.l2 = layers_size[2]
        self.l3 = layers_size[3]
        
        self.weights = {'l1': np.random.randn(self.l1, self.l0)/np.sqrt(self.l1),
                        'l2': np.random.randn(self.l2, self.l1)/np.sqrt(self.l2),
                        'l3': np.random.randn(self.l3, self.l2)/np.sqrt(self.l3) }

        self.biases = { 'l1': np.random.randn(1, self.l1),
                        'l2': np.random.randn(1, self.l2),
                        'l3': np.random.randn(1, self.l3) }

        # for viewing network structure in console
        print("\n[network]: Intializing network with ...\n",104*"-")
        print("#nodes: layer0[input]:",self.l0," \t| layer1: ",self.l1, " \t| layer2: ",self.l2,               " \t\t| layer3[out]: ",self.l3)
        print("weights: \t\t","- \t| layer1: ",self.weights['l1'].shape," \t| layer2: ",self.weights['l2'].shape," \t| layer3[out]: ",self.weights['l3'].shape)
        print("biases: \t\t","- \t| layer1: ",self.biases['l1'].shape,  " \t| layer2: ",self.biases['l2'].shape, " \t| layer3[out]: ",self.biases['l3'].shape)
        print(105*"-","\n")
    
    def fit(self, X, y, ephocs):

        """
        this method is an implementation training network by using Stocastic-gradient-decent (SGD),
        it was done by these steps
            1. forward propagation              -->|-->|-->|-->
            2. backward propagation             <--|<--|<--|<--    
            3. updating weights and biases      -+-+-+-+-+-+-+-
        
        Parameters:
            X   :   numpy array     (contains all features)
            y   :   numpy array     (contains all targets)
        
        Returns:
            (no-returns)

        """

        print("[network]: Training network on ",ephocs," ephocs.")

        if(self.validate):
            self.X_val  = X[40001:]        # out of 42000 samples we taking 2000 (X and y)
            self.y_val  = y[40001:]        # for validation

            X = X[:40000]                   # updating training set
            y = y[:40000]                   # so that we can't see test sapmples in validation set

        batch_size = X.shape[0]
        
        for ephoch in range(ephocs):

            total_error = []

            # shuffling
            if(self.isShuffle):
                util.shuffle(X, y)    

            for i in range(batch_size):

                # feed-forward
                zs, activations = self.__forward__(X[i])
                activations.append(y[i])

                # back-forward
                error, new_weights, new_biases = self.__backward__(zs, activations, y[i])
                total_error.append(np.mean(np.abs(error)))

                # weights & biases update
                self.weights['l3'] -= self.learning_rate * new_weights[0]
                self.weights['l2'] -= self.learning_rate * new_weights[1]
                self.weights['l1'] -= self.learning_rate * new_weights[2]

                self.biases['l3'] -= self.learning_rate * new_biases[0]
                self.biases['l2'] -= self.learning_rate * new_biases[1]
                self.biases['l1'] -= self.learning_rate * new_biases[2]
            
            if(self.isValidate):
                print("\t   ephoch ",(ephoch+1),"\t...\t train_loss:", round((sum(total_error)/batch_size)*100,2),"%",
                                                     "\t val_loss:", round(self.validate(self.X_val, self.y_val)*100,2),"%")
            else:
                print("\t   ephoch ",(ephoch+1),"\t...\t train_loss:",round((sum(total_error)/batch_size)*100,2),"%")
        
        print("\n")
    

    
    def __forward__(self, x):

        """
        this method is an implementation of forward propagation with one sample at a time.
        
        Parameters: 
            x          :   numpy array  (contains one sample of features)
                    
        Returns:
            zs         :    list        (contains numpy arrays, each array coresponds to sum(xW+b) of respective layer)
            activations:    list        (contains numpy arrays, each array coresponds to output of respective layer)
        
        """
                                                                      # demo shapes
        l0 = x.T                                                      # [1, 784]
        z1 = np.dot(l0, self.weights['l1'].T) + self.biases['l1']     # [1, 300] = [1, 784] .* [784, 300] + [1, 300]
        l1 = util.relu(z1)                                            # [1, 300]
        
        z2 = np.dot(l1, self.weights['l2'].T) + self.biases['l2']     # [1, 90]  = [1, 300] .* [300, 90] + [1, 90] 
        l2 = util.relu(z2)                                            # [1, 90]
        
        z3 = np.dot(l2, self.weights['l3'].T) + self.biases['l3']     # [1, 10]  = [1, 90] .* [90, 10] + [1, 10]
        l3 = util.softmax(z3)                                         # [1, 10]

        zs = [z1, z2, z3]
        activations = [l0, l1, l2, l3]

        return zs, activations
    

    def __backward__(self, zs, activations, y):
        """
        this method is an implementation of backpropagation with one sample at a time.
        
        Parameters:
            zs         : list        (contains numpy arrays, each array coresponds to sum(xW+b) of respective layer) 
            activations: list           (contains numpy arrays, each array coresponds to output of respective layer)
            y          : numpy array    (contains one sample of target values)

        Returns:    
            l3_error   : numpy array    (contains error of last layer 3 (or) network error)
            new_weights: numpy array    (contains numpy arrays, each array coresponds to new weights of respective layer)
            new_biases : numpy array    (contains numpy arrays, each array coresponds to new biases of respective layer)
        """
        
        l0, l1, l2, l3 = activations[0], activations[1], activations[2], activations[3]
        z1, z2 = zs[0], zs[1]

        # calculating loss of network (or) layer 3           # demo shapes
        l3_error = l3 - y.T                                  # [1, 10] = [1, 10] - [1, 10]

        # calculating  layer3 weights and biases
        l3_delta = np.multiply(l3_error, l3)                 # [1, 10] = [1, 0] * [1, 10]
        l3_new_weights = l3_delta.T.dot(l2)                  # [10, 90] = [10, 1] * [1,90]
        
        # calculating  layer2 weights and biases
        l2_error = l3_delta.dot(self.weights['l3'])          # [1, 90] = [1, 10] * [10, 90]
        l2_delta = np.multiply(l2_error, util.relu(z2, derivative=True)) # [1, 90] = [1, 90] * [1,90]
        l2_new_weights = l2_delta.T.dot(l1)                  # [90, 300] = [90, 1] * [1, 300]
        
        # calculating  layer1 weights and biases
        l1_error = l2_delta.dot(self.weights['l2'])          # [1, 300] = [1, 90] * [90, 300]
        l1_delta = np.multiply(l1_error, util.relu(z1, derivative=True)) # [1,300] = [1, 300] * [1, 300]
        l1_new_weights = l1_delta.T.dot(l0)                  # [300, 784] = [300, 1] * [1, 784]

        new_weights = [l3_new_weights, l2_new_weights, l1_new_weights]
        new_biases = [l3_delta, l2_delta, l1_delta]

        return l3_error, new_weights, new_biases


    def predict(self, x, show=False):
        """
        this method is resposible to predict the digit by using network weights and biases.

        Parameters:
            x   : numpy array       (contains one sample of input features)
            show: boolean           (boolean flag for displaying given sample 'x' into the screen)  

        Returns:
            integer                 (integer is an network predicted digit number) 
        """

        _ , activations = self.__forward__(x)
        
        if(show):
            util.view_digit(x.T, str(np.argmax(activations[-1])))
        
        return np.argmax(activations[-1])
    

    def validate(self, x, y):
        """
        this method is resposible to validate the network performance by using validation data.

        Parameters:
            x   : numpy array       (contains validate features)
            y   : numpy array       (contains validate targets)  

        Returns:
            integer                 (integer is an validation loss (or) how many samples network predicted currectly in test data) 
        """

        val_score = []
        for i in range(x.shape[0]):

            _ , activations = self.__forward__(x[i])
            val_score.append(np.mean(np.abs(y[i].T - activations[-1])))
            
        return sum(val_score)/x.shape[0]