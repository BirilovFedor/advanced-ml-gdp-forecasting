from Models.CNN_LSTM import CNN_LSTM
from Models.PytorchLSTM import Pytorch_LSTM
import torch.nn as nn
import torch.optim as optim
import itertools
import pandas as pd
import matplotlib.pyplot as plt

class CrossValidationAutoencoder:

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

    def run_cross_validation(self, cross_val_parameter: dict):

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

        total_losses, recon_losses, kl_losses = [], [], []

        for i, parameter in enumerate(cross_val_parameter[key], 1):

            parameter_dict = {key: parameter}
            print(f"[{i}/{len(cross_val_parameter[key])}] {parameter_dict}")

            training_params = {**self.training_fixed_params,
                               **{k: v for k, v in parameter_dict.items() if k in self.training_param_keys}}
            model_params    = {**self.model_fixed_params,
                               **{k: v for k, v in parameter_dict.items() if k not in self.training_param_keys}}
            
            model = self.model_class(**model_params)

            tl, rl, kl = model.fit(train_data = self.ts.torch_flattened_training_data,
                                    num_epoches = training_params["num_epochs"],
                                    batch_size  = training_params["batch_size"],
                                    optimizer   = self.optimizer(model.parameters(), lr=training_params["learning_rate"]))
            total_losses.append(tl[-1])
            recon_losses.append(rl[-1])
            kl_losses.append(kl[-1])

        plt.figure(figsize=(8,6), dpi=600)
        plt.plot(cross_val_parameter[key], total_losses, color = "red", label = "total loss")
        plt.plot(cross_val_parameter[key], recon_losses, color="blue", label = "reconstruction loss")
        plt.plot(cross_val_parameter[key], kl_losses, color="green", label = "KL divergence")
        plt.xlabel(key)
        plt.ylabel('Loss Criteria')
        plt.legend()
        #plt.show()
        plt.savefig(f'/Users/arina/Documents/results/Autoencoder/Cross-Validation Results Over {key}')