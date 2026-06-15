import numpy as np   
np.random.seed(42)               


class MLP:

    '''
    Docstring for MLP

    Class that is used for Multi Layer Perceptron Application
    It should be initialized with an array of a size n that contain the following information if order
    0: number of features in the vector of the initial data (number of features for X)
    1-(n-1): number of perceptrons at each layer of n - 1 hidden layers
    n: number of perceptron on the output layer

    '''

    def __init__(self, layer_sizes):

        '''
        Docstring for __init__
        
        :param self: Description
        :param layer_sizes: array of a size n that contain number of perceptron for each layer
        :param weigths: array that keeps matrixes of weigths for each layer
        :param bias: array that keeps matrixes of biases for each layer
        :param z: array that keeps preactivation function results for each layer
        :param a: array that keeps activation function results for each layer
        :param sigma: array that keeps error terms for each layer
        :param wd: array that keeps loss derivative agains weight for each layer
        :param bd: array that keeps loss derivative agains bias for each layer
        '''

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights, self.bias = self.get_weights_biases()
        self.z = [0]*self.num_layers
        self.a = [0]*self.num_layers
        self.sigma = [0]*self.num_layers
        self.wd = [0]*self.num_layers
        self.bd = [0]*self.num_layers


    def get_weights_biases(self):

        # initialize the array with weigths and biases for each of the layers

        '''
        Docstring for get_weights_biases

        function is generating a random ~N(0, 1) numpy arrays for weight and bias
        based on the layer sizes provided in initialization array 
        
        :param self: Description

        :return weights: list of numpy arrays, where each array is weigths for a layer
        :return bias:  list of numpy vectors, where each vector is bias for a layer
        '''

        weights = []
        bias = []


        for i in range(self.num_layers):

            W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            b = np.zeros((1, self.layer_sizes[i+1]))

            weights.append(W)
            bias.append(b)

        return weights, bias
    
    def preactivation_function(self, a, w, b):
        # function is used to calculate z(l) = w(l)*a(l-1)+b in a matrix form

        return np.dot(a, w) + b
    
    def relu_activation_function(self, z):
        # activation function max(0, z)
        return np.maximum(0, z)
    
    def relu_activation_function_derivative(self, z):
        # derivative of a relu activation function, for each value in z point 0 if z<=0 and 1 if z>0
        return (z > 0).astype(float)
        
    def softmax_activation_function(self, z):
        # calculate the softmax activation function e^z/sum(e^z)

        stable_z = z - np.max(z, axis = -1, keepdims=True)
        exp_z = np.exp(stable_z)
        return exp_z/np.sum(exp_z, axis=-1, keepdims=True)
    
    def CE_function_derivative(self, y_pred, y_true):
        # calculate the y_hat - y
        y_true_onehot_encoded = np.eye(self.layer_sizes[len(self.layer_sizes) - 1])[y_true]
        return y_pred - y_true_onehot_encoded
    
    def forward_pass(self, X):
        # calculate steps for each layer:
        # step one calculate preactivation function: z(l) = w(l)*a(l-1)+b
        # step two calculate activation function: a(l) = g(z(l))

        '''
        Docstring for forward_pass
        
        this function is a realization of a forward pass for a MLP model. The following calculations are
        apllied for each leayer:

        step one calculate preactivation function: z(l) = w(l)*a(l-1)+b
        step two calculate activation function: a(l) = g(z(l))


        :param self: Description
        :param X: the initial batch of data, presented as an array
        '''

        # layer 1
        self.z[0] = self.preactivation_function(X, self.weights[0], self.bias[0])
        self.a[0] = self.relu_activation_function(self.z[0])

        # hidden layers
        for layer in range(1, self.num_layers - 1):
            self.z[layer] = self.preactivation_function(self.a[layer-1], self.weights[layer], self.bias[layer])
            self.a[layer] = self.relu_activation_function(self.z[layer])

        # output layer
        self.z[self.num_layers - 1] = self.preactivation_function(self.a[self.num_layers-2], self.weights[self.num_layers-1], self.bias[self.num_layers-1])
        self.a[self.num_layers - 1] = self.softmax_activation_function(self.z[self.num_layers-1])
    
    def backpropagation(self, X, y_true):

        '''
        Docstring for backpropagation

        This function is a realization of a backpropagation step of MLP model. 
        
        :param self: Description
        :param y_true: actual repsonse variable

        '''

        # number of batches
        N = X.shape[0]
        
        self.sigma[self.num_layers - 1] = self.CE_function_derivative(y_pred=self.a[self.num_layers-1], y_true=y_true)
        self.wd[self.num_layers - 1] = np.dot(self.a[self.num_layers - 2].T, self.sigma[self.num_layers - 1]) / N
        self.bd[self.num_layers - 1] = np.mean(self.sigma[self.num_layers - 1], axis=0, keepdims=True)
        
        for iter in range(self.num_layers - 2, 0, -1):
            self.sigma[iter] = np.dot(self.sigma[iter+1], self.weights[iter+1].T) * self.relu_activation_function_derivative(self.z[iter])
            self.wd[iter] = np.dot(self.a[iter-1].T, self.sigma[iter]) / N
            self.bd[iter] = np.mean(self.sigma[iter], axis=0, keepdims=True)

        self.sigma[0] = np.dot(self.sigma[1], self.weights[1].T) * self.relu_activation_function_derivative(self.z[0])
        self.wd[0] = np.dot(X.T, self.sigma[0]) / N
        self.bd[0] = np.mean(self.sigma[0], axis=0, keepdims=True)

    def update_weights(self, learning_rate):

        '''
        Docstring for update_weights

        This function is desined to updated weights and biases using the gradient descent:

        w_i = w_i-1 - λ*dL/dwi-1
        b_i = b_i-1 - λ*dL/dbi-1
        
        :param self: Description
        :param learning_rate: rate which defines the lenght of a step along the calculated derivative
        (bigger the batch size, bigger the step)

        '''

        for iter in range(self.num_layers):
            self.weights[iter] -= learning_rate * self.wd[iter]
            self.bias[iter] -= learning_rate * self.bd[iter]

    def training(self, X, y, X_test, y_test, learning_rate = 0.1, batch_size = 10, num_epoches = 100):
        
        '''
        Docstring for training
        
        :param self: Description
        :param X: numpy matrix N x M that contain vector embeddings,
        where N is number of countries and M is number of featues in the embedding
        :param y: list of a size 1 x N, which contain corresponding y values for X
        :param learning_rate: learning rate which used to decide how far are we moving along the derivative 
        during the gradient descent step
        :param batch_size: number of countries feed to the model at each training iteration
        :param num_epoches: this variable describe number of epoches that should be used in training
        '''

        steps_counter = 0

        self.training_accuracy_per_epoch = []
        self.testing_accuracy_per_epoch = []

        for epoch in range(num_epoches):

            # shaffle the data for each epoch to improve training
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_batch_accuracies = []

            # iterate over the whole dataset with a ste pof a batch size
            for start in range(0, X.shape[0], batch_size):


                # take batch from the original dataset
                batch_X = X_shuffled[start:start+batch_size]
                batch_y = y_shuffled[start:start+batch_size]

                self.forward_pass(batch_X)

                self.backpropagation(batch_X, batch_y)

                self.update_weights(learning_rate=learning_rate)

                batch_accuracy = self.compute_accuracy(self.a[self.num_layers - 1], batch_y)

                epoch_batch_accuracies.append(batch_accuracy)

                steps_counter+=1
            
            self.training_accuracy_per_epoch.append(np.mean(epoch_batch_accuracies))

            test_predictions = self.predict(X_test=X_test)
            self.testing_accuracy_per_epoch.append(np.mean(test_predictions==y_test))

        print(f"Training is successfully done with {steps_counter} number of steps")

    def compute_accuracy(self, y_pred, y_true):

        '''
        Docstring for compute_accuracy

        Computes the classification accuracy for a batch by comparing the
        predicted class (argmax of softmax output) against the true labels.

        :param self: Description
        :param y_pred: numpy array of shape (N, num_classes) with softmax probabilities
        :param y_true: numpy array of shape (N,) with true class indices

        :return accuracy: float, fraction of correctly classified samples

        '''
        predicted_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predicted_classes == y_true)
        return accuracy
    
    def compute_f1(self, y_pred, y_true):

        '''
        Docstring for compute_f1 

        Computes the macro-average classification f1 score for each class and average value for a batch by comparing the
        predicted class (argmax of softmax output) against the true labels.

        :param self: Description
        :param y_pred: numpy array of shape (N, num_classes) with softmax probabilities
        :param y_true: numpy array of shape (N,) with true class indices

        :return f1_scores: array that contain f1_score for each of the classes [f1 for class1, f1 for class2, ....]
        :return macro_f1: single number average of f1 scores across all classes
        '''


        num_classes = self.layer_sizes[-1]
        f1_scores = []

        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores)
        return f1_scores, macro_f1
    
    def testing(self, X_test, y_test, is_f1=False):

        '''
        Docstring for testing

        :param X_test: features of testing dataset
        :param y_test: responses of testing dataset
        :param is_f1: to decide if the f1 score should be returned or not
        '''


        test_predictions = self.predict(X_test=X_test)
        if is_f1:
            f1_scores, macro_f1_score = self.compute_f1(test_predictions, y_test)
            return np.mean(test_predictions==y_test), f1_scores, macro_f1_score
        return np.mean(test_predictions==y_test)
    
    def predict(self, X_test):

        '''

        Docstring for predict

        function is used to make a prediction, using the weights and bias from the training

        :param X_test: where class prediction is required

        '''

        # perform a forward pass of a model
        self.forward_pass(X_test)

        return np.argmax(self.a[-1], axis=1)

    def compute_metrics(self, y_true, X_test):
        num_classes = self.layer_sizes[-1]
        y_pred = self.predict(X_test)

        accuracy = np.mean(y_pred == y_true)

        precision_per_class = []
        recall_per_class = []
        specificity_per_class = []
        f1_per_class = []

        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            tn = np.sum((y_pred != c) & (y_true != c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))


            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            specificity_per_class.append(specificity)
            f1_per_class.append(f1)

        print('Evaluation metrics of optimal MLP:')
        print('Accuracy:', accuracy)
        print('F1 score', np.mean(f1_per_class))

        for c in range(num_classes):

            print(f'\nPrecision class {c}:', precision_per_class[c])
            print(f'Recall class {c}:', recall_per_class[c])
            print(f'Specificity class {c}:', specificity_per_class[c])
            print(f'F1-Score class {c}:', f1_per_class[c])


        #return accuracy, precision_per_class, recall_per_class, specificity_per_class, f1_per_class



