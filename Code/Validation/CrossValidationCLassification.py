from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

class CrossValidationClassification:

    def __init__(self, model_class, model_fixed_params, training_fixed_params,
                 X, y, training_param_keys: list):
        '''
        Docstring for __init__

        :param model_class: class of a predefined Pytorch model
        :param model_fixed_params: fixed parameters that used in __init__() of a model class
        :param training_fixed_params: fixed training parameters that used in fit() of a model class
        :param X: training data in original scale
        :param y: response for the training data
        :param training_param_keys: names for all training parameters that are used in fit() funciton
        '''
        self.model_class            = model_class
        self.model_fixed_params     = model_fixed_params
        self.training_fixed_params  = training_fixed_params
        self.X                     = X
        self.y                      = y
        self.training_param_keys    = set(training_param_keys)

    def run_cross_validation(self, cross_val_parameter: dict, n_splits=5, random_state=42, file_name =''):

        '''
        Docstring for run_cross_validation

        :param cross_val_parameter: dictionary that can store only single key - name of a 
        parameter to cross-validate for, and corresponding valued of this parameter
        parameter defined in this dictionary will overwrite the constant parameter 
        defined in __init__() function
        

        funcntion is performing cross-validation over given results and plot the results
        for training and validation loss metric
        '''

        key = list(cross_val_parameter.keys())[0]

        val_accuracy = []
        val_f1_agregated = []
        val_f1_0 = []
        val_f1_1 = []
        val_f1_2 = []
        val_f1_3 = []

        for i, parameter in enumerate(cross_val_parameter[key], 1):
            
            parameter_dict = {key: parameter}
            print(f"[{i}/{len(cross_val_parameter[key])}] {parameter_dict}")

            training_params = {**self.training_fixed_params,
                               **{k: v for k, v in parameter_dict.items() if k in self.training_param_keys}}
            model_params    = {**self.model_fixed_params,
                               **{k: v for k, v in parameter_dict.items() if k not in self.training_param_keys}}
            
            split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            scaler = StandardScaler()

            

            val_accuracy_par = []
            val_f1_agregated_par = []
            val_f1_0_par = []
            val_f1_1_par = []
            val_f1_2_par = []
            val_f1_3_par = []
            for train_index, validation_index in split.split(self.X, self.y):
                
                model = self.model_class(**model_params)

                X_train, y_train = self.X[train_index].copy(), self.y[train_index].copy()
                X_val, y_val = self.X[validation_index].copy(), self.y[validation_index].copy()

                scaled_X_train = scaler.fit_transform(X_train)
                scaled_X_val = scaler.transform(X_val)

                model.training( X=scaled_X_train, y=y_train, 
                                                   X_test=scaled_X_val, y_test=y_val, 
                                                   learning_rate = training_params["learning_rate"], 
                                                   batch_size = training_params["batch_size"], 
                                                   num_epoches = training_params["num_epochs"])
                
                accuracy, f1_scores, macro_f1_score =  model.testing(X_test=scaled_X_val, 
                                                                     y_test=y_val, is_f1=True)
                
                val_accuracy_par.append(accuracy)
                val_f1_agregated_par.append(macro_f1_score)
                val_f1_0_par.append(f1_scores[0])
                val_f1_1_par.append(f1_scores[1])
                val_f1_2_par.append(f1_scores[2])
                val_f1_3_par.append(f1_scores[3])
        
            val_accuracy.append(np.mean(val_accuracy_par))
            val_f1_agregated.append(np.mean(val_f1_agregated_par))
            val_f1_0.append(np.mean(val_f1_0_par))
            val_f1_1.append(np.mean(val_f1_1_par))
            val_f1_2.append(np.mean(val_f1_2_par))
            val_f1_3.append(np.mean(val_f1_3_par))

        
        plt.figure(figsize=(8,6))
        plt.plot(cross_val_parameter[key], val_accuracy, color='blue', label='Cross-Validation Accuracy')
        plt.plot(cross_val_parameter[key], val_f1_agregated, color='green', label='Cross-Validation F1-Score')
        plt.plot(cross_val_parameter[key], val_f1_0, color='orange', label='Cross-Validation F1-Score (class 0)')
        plt.plot(cross_val_parameter[key], val_f1_1, color='purple', label='Cross-Validation F1-Score (class 1)')
        plt.plot(cross_val_parameter[key], val_f1_2, color='pink', label='Cross-Validation F1-Score (class 2)')
        plt.plot(cross_val_parameter[key], val_f1_3, color='brown', label='Cross-Validation F1-Score (class 3)')
        plt.title(f'Evaluation Across {key}')
        plt.xlabel(key)
        plt.ylabel('Evaluation Metrics')
        plt.grid()
        plt.legend()
        if file_name:
            plt.savefig(file_name, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()