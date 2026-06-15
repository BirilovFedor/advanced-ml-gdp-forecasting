from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import itertools
np.random.seed(42)     
class GridSearchClassification:

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

    def run_grid_search(self, param_grid, top_n = 10, n_splits=5, random_state=42):

        '''
        Docstring for run_grid_search

        :param param_grid: dictionary that store the parameters to search over,
        keys are the names of parameters, values are the array with values to search over,
        all parameters defined in this dictionary will overwrite the constant parameters 
        defined in __init__() function
        :param top_n: parameter to define number of top models to print out, predefined to be 10
        

        function is searching over all possible subsets of features, and show best model by validation results
        also showing the training loss metric
        '''

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values())) 

        print(f"{'='*55}")
        print(f"  {self.model_class.__name__} — Grid Search")
        print(f"  {len(combos)} combinations to evaluate")
        print(f"{'='*55}\n")

        overall_results = []

        
        for i, combo in enumerate(combos, 1):

            parameters = dict(zip(keys, combo))
            print(f"[{i}/{len(combos)}] {parameters}")

            training_params = {**self.training_fixed_params,
                               **{k: v for k, v in parameters.items() if k in self.training_param_keys}}
            model_params    = {**self.model_fixed_params,
                               **{k: v for k, v in parameters.items() if k not in self.training_param_keys}}
            
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

            row  = {**parameters, "cross_val_acc":np.mean(val_accuracy_par) ,
                      "cross_agr_f1": np.mean(val_f1_agregated_par),
                      "cross_f1_0": np.mean(val_f1_0_par),
                      "cross_f1_1": np.mean(val_f1_1_par),
                      "cross_f1_2": np.mean(val_f1_2_par),
                      "cross_f1_3": np.mean(val_f1_3_par)}
            overall_results.append(row)

        results_df = (pd.DataFrame(overall_results).sort_values("cross_val_acc", ascending=False).reset_index(drop=True))

        print(f"{'='*55}")
        print(f"  Top {top_n} Configurations by Cross-Validation Loss Metric")
        print(f"{'='*55}")
        print(results_df.head(top_n).to_string(index=False))
        return results_df