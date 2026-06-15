import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools

class GridSearchTimeSeries:

    def __init__(self, model_class, model_fixed_params, training_fixed_params,
                 ts, training_param_keys: list,
                 optimizer=optim.Adam, criterion=nn.L1Loss()):
        
        '''
        Docstring for __init__

        :param model_class: class of a predefined Pytorch model
        :param model_fixed_params: fixed parameters that used in __init__() of a model class
        :param training_fixed_params: fixed training parameters that used in fit() of a model class
        :param ts: object of a TimeSeries object that stores the data
        :param training_param_keys: names for all training parameters that are used in fit() funciton
        :param optimizer: optimizer, set to optim.Adam as default
        :param criterion: loss criteion used to update the model, set to L1Loss by default
        '''
        self.model_class            = model_class
        self.model_fixed_params     = model_fixed_params
        self.training_fixed_params  = training_fixed_params
        self.ts                     = ts
        self.training_param_keys    = set(training_param_keys)
        self.optimizer              = optimizer
        self.criterion              = criterion

    def run_grid_search(self, param_grid, top_n = 10):

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
            
            model = self.model_class(**model_params)

            train_loss = model.fit(train_data = self.ts.torch_training_time_series,
                                    num_epoches = training_params["num_epochs"],
                                    batch_size  = training_params["batch_size"],
                                    optimizer   = self.optimizer(model.parameters(), lr=training_params["learning_rate"]),
                                    criterion   = self.criterion,
                                    is_return   = True)
            
            val_loss = model.evaluate(
                data       = self.ts.torch_validation_time_series,
                batch_size = training_params["batch_size"],
                criterion  = self.criterion,
            )

            row   = {**parameters, "train_score": train_loss, "val_score": val_loss}
            overall_results.append(row)

        results_df = (pd.DataFrame(overall_results).sort_values("val_score").reset_index(drop=True))

        print(f"{'='*55}")
        print(f"  Top {top_n} Configurations by Validation L1")
        print(f"{'='*55}")
        print(results_df.head(top_n).to_string(index=False))
        return results_df