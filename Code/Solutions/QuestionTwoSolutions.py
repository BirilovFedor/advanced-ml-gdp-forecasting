# ──────────────────────── import from the project ──────────────────────────────────────────────────
from Validation.CrossValidationCLassification import CrossValidationClassification
from Validation.GridSearchClassification import GridSearchClassification
from Models.MLP import MLP

# ──────────────────────── import from external libraries ──────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class QuestionTwoSolutions:

    def __init__(self, dataset, features):
        '''
        Docstring for __init__

        :param dataset; instance of Dataset class that contain all the sequences
        :param features: numerical features that are used in the analysis process

        '''

        self.dataset = dataset
        self.features = features

    def solve_question(self):
        '''
        Docstring for solve_question

        runner for question two
        please uncomment the function you want to be runned

        '''

        # ────────────────────────  Create embedding with log transformed gdp binning ───────────────────────────────────────────────────
        self.training_embedding, self.testing_embedding = QuestionTwoSolutions.get_labbeled_embeddings(self.dataset)

        # ────────────────────────  Get train test split  ───────────────────────────────────────────────────
        self.X_train, self.y_train, self.X_test, self.y_test = \
                QuestionTwoSolutions.get_train_test_split(self.training_embedding, self.testing_embedding)
        
        # ────────────────────────  Scale the data ───────────────────────────────────────────────────
        self.scaled_X_train, self.scaled_X_test, self.scaler = \
                QuestionTwoSolutions.normalize_data(self.X_train, self.X_test)
        
        # ────────────────────────  Grid Search ───────────────────────────────────────────────────
        '''QuestionTwoSolutions.grid_search(self.X_train, self.y_train)'''

        # ────────────────────────  Cross-Validation over batch size ───────────────────────────────────────────────────
        '''QuestionTwoSolutions.cross_validation(self.X_train, 
                                              self.y_train,
                                                layer_sizes= [self.X_train.shape[1], 64, 32, 16, 4],
                                                batch_size=5,
                                                learning_rate=0.001,
                                                num_epochs=400,
                                                validation_dict={"batch_size":[5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 140, 200]})'''
        
        # ────────────────────────  Cross-Validation over number of epochs ───────────────────────────────────────────────────
        '''QuestionTwoSolutions.cross_validation(self.X_train, 
                                                self.y_train,
                                                layer_sizes= [self.X_train.shape[1], 64, 32, 16, 4],
                                                batch_size=5,
                                                learning_rate=0.001,
                                                num_epochs=400,
                                                validation_dict={"num_epochs":[5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 800]})'''
        
        # ────────────────────────  Cross-Validation over learning rate ───────────────────────────────────────────────────
        '''QuestionTwoSolutions.cross_validation(self.X_train, 
                                                self.y_train,
                                                layer_sizes= [self.X_train.shape[1], 64, 32, 16, 4],
                                                batch_size=5,
                                                learning_rate=0.001,
                                                num_epochs=400,
                                                validation_dict={"learning_rate":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]})'''
        
        # ────────────────────────  Fit MLP plot testing vs training for best grid search model ───────────────────────────────────────────────────
        '''QuestionTwoSolutions.fit_MLP(scaled_X_train=self.scaled_X_train,
                                     y_train=self.y_train,
                                     scaled_X_test=self.scaled_X_test,
                                     y_test=self.y_test,
                                     layer_sizes=[self.scaled_X_train.shape[1], 64, 32, 16, 4],
                                     batch_size=100, learning_rate=0.1, num_epoches=300,
                                     file_name='/Users/arina/Documents/results/MLP/Testing_vs_training_accuracy_1.png')'''
        
        
        # ────────────────────────  Fit MLP plot testing vs training for best CV model ───────────────────────────────────────────────────
        '''np.random.seed(42)  
        QuestionTwoSolutions.fit_MLP(scaled_X_train=self.scaled_X_train,
                                     y_train=self.y_train,
                                     scaled_X_test=self.scaled_X_test,
                                     y_test=self.y_test,
                                     layer_sizes=[self.scaled_X_train.shape[1], 64, 32, 16, 4],
                                     batch_size=5, learning_rate=0.001, num_epoches=400,
                                     file_name='/Users/arina/Documents/results/MLP/Testing_vs_training_accuracy_2.png')'''
        
        # ────────────────────────  Evaluate Performance on testing data ───────────────────────────────────────────────────
        QuestionTwoSolutions.evaluate_model_preformance(scaled_X_train=self.scaled_X_train,
                                                        y_train=self.y_train,
                                                        scaled_X_test=self.scaled_X_test,
                                                        y_test=self.y_test,
                                                        layer_sizes=[self.scaled_X_train.shape[1], 64, 32, 16, 4],
                                                        batch_size=5, learning_rate=0.001, num_epoches=400)
 
    @staticmethod
    def get_labbeled_embeddings(dataset):
        '''
        Docstring for get_labbeled_embeddings

        :param dataset: instance of Dataset class

        Function is creating embedding, applying the log gdp binning to get classes for the data

        :return the training and testing dataframes that contain country, embedding and the corresponding class
        '''

        dataset.get_embedding()

        return dataset.training_embedding, dataset.testing_embedding
    
    @staticmethod
    def get_train_test_split(training_embedding, testing_embedding):
        '''
        Docstring for get_train_test_split

        :param training_embedding: dataframe with training embeddings and responses
        :param testing_embedding: dataframe with testing embeddings and responses

        : return X_train, y_train, X_test and y_test that are ready to be used
        '''

        X_training = np.vstack(training_embedding['embedding'].to_list())
        X_testing = np.vstack(testing_embedding['embedding'].to_list())

        y_training = np.array(training_embedding['label'].to_list())
        y_testing = np.array(testing_embedding['label'].to_list())
        return X_training, y_training, X_testing, y_testing

    @staticmethod
    def normalize_data(X_training, X_testing):
        '''
        Docstring for normalize_data

        :param X_training: training data
        :param X_testing: testing data

        : return scaled  X_train and  X_test and scaler for inverce scaling
        '''

        #create scaler instance
        scaler = StandardScaler()
        # fit scaler on training data and transform training data
        scaled_X_training = scaler.fit_transform(X_training)
        # transform only the testing data
        scaled_X_testing = scaler.transform(X_testing)

        return scaled_X_training, scaled_X_testing, scaler
    
    @staticmethod
    def cross_validation(X_train, y_train,layer_sizes, batch_size, learning_rate, num_epochs, validation_dict):

        '''
        Docstring for cross_validation

        :param X_train: training data, not scaled
        :param y_train: trainign data responses
        :param layer_sizes: array with all layer size to identify MLP model
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param num_epochs: number of epochs
        :param validation_dict: dictionary with a single elment where the key is the name of variable to cross-validate over
        and the value for this key is array with value for cross-validation
        '''

        validator = CrossValidationClassification(model_class=MLP, 
                                                    model_fixed_params={
                                                        "layer_sizes": layer_sizes
                                                    },
                                                    training_fixed_params={
                                                        "batch_size":    batch_size,
                                                        "learning_rate": learning_rate,
                                                        "num_epochs":   num_epochs,
                                                    },
                                                    X=X_train,
                                                    y=y_train,
                                                    training_param_keys = ["batch_size", "learning_rate", "num_epochs"])
        validator.run_cross_validation(cross_val_parameter=validation_dict, 
                                       file_name=f'/Users/arina/Documents/results/MLP/Cross-Validation Results Over {list(validation_dict.keys())[0]}')

    @staticmethod
    def grid_search(X_train, y_train):

        '''
        Docstring for grid_search

        :param X_train: training data, not scaled
        :param y_train: trainign data responses

        Grid seach over parameter space, you ahve to update the search space inside of a function
        The top n models and their specifications will be printed for further investigation
        '''

        grid_searcher = GridSearchClassification(model_class=MLP, 
                                                    model_fixed_params={
                                                        "layer_sizes": [X_train.shape[1], 128, 64, 32, 4]
                                                    },
                                                    training_fixed_params={
                                                        "batch_size":    10,
                                                        "learning_rate": 0.01,
                                                        "num_epochs":   100,
                                                    },
                                                    X=X_train,
                                                    y=y_train,
                                                    training_param_keys = ["batch_size", "learning_rate", "num_epochs"])
        
        results = grid_searcher.run_grid_search(param_grid={
            'layer_sizes': [(X_train.shape[1], 32, 4), (X_train.shape[1], 64, 4), (X_train.shape[1], 64, 32, 4), (X_train.shape[1], 128, 64, 4), (X_train.shape[1], 128,64,32, 4), (X_train.shape[1], 64,32,16, 4)],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'batch_size': [5, 10, 15, 20, 50, 100],
            'num_epochs': [20, 40, 60, 80, 100, 200, 300, 400, 600 ]}, top_n = 10)

    @staticmethod
    def fit_MLP(scaled_X_train, y_train, scaled_X_test, y_test, layer_sizes, batch_size, learning_rate, num_epoches, file_name=''):
        
        '''
        Docstring for fit_MLP

        :param scaled_X_train: training data,  scaled
        :param y_train: training data responses
        :param scaled_X_test: training data, not scaled
        :param y_test: testing data responses
        :param layer_sizes: array with all layer size to identify MLP model
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param num_epochs: number of epochs
        :param file_name: name of a file where to save the plot, optional

        fit MLP and plot traing vs test across batches
        '''

        mlp_model = MLP(layer_sizes)

        mlp_model.training(scaled_X_train, y_train, scaled_X_test, y_test, batch_size=batch_size, learning_rate=learning_rate, num_epoches=num_epoches)


        plt.figure(figsize=(8,6))
        plt.plot(range(1, len(mlp_model.training_accuracy_per_epoch) + 1), mlp_model.training_accuracy_per_epoch, color='blue', label='Training Accuracy')
        plt.plot(range(1, len(mlp_model.testing_accuracy_per_epoch) + 1), mlp_model.testing_accuracy_per_epoch, color='red', label='Testing Accuracy')
        plt.title('Training vs. Testing Accuracy of the MLP Model')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Epochs')
        plt.legend()
        if file_name:
            plt.savefig(file_name, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def evaluate_model_preformance(layer_sizes, scaled_X_train, y_train, scaled_X_test, y_test, batch_size, learning_rate, num_epoches):

        '''
        Docstring for evaluate_model_preformance

        :param scaled_X_train: training data,  scaled
        :param y_train: training data responses
        :param scaled_X_test: training data, not scaled
        :param y_test: testing data responses
        :param layer_sizes: array with all layer size to identify MLP model
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param num_epochs: number of epochs
        
        fit MLP and evaludate performance on a testing dataset including f1 scores, accuracy, precision, recall and specificaty
        all results will be printed out
        '''
        
        mlp_model = MLP(layer_sizes)

        mlp_model.training(scaled_X_train, y_train, scaled_X_test, y_test, batch_size=batch_size, learning_rate=learning_rate, num_epoches=num_epoches)
        mlp_model.compute_metrics(y_test, scaled_X_test)