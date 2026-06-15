# ──────────────────────── import from the project ──────────────────────────────────────────────────
from DataStorage.TimeSeries import TimeSeries
from Models.PytorchLSTM import Pytorch_LSTM
from Models.CNN_LSTM import CNN_LSTM
from Models.Transformer import Transformer
from Validation.TimeSeriesCrossValidation import TimeSeriesCrossValidation
from Validation.GridSearchTimeSeries import GridSearchTimeSeries



# ──────────────────────── import from external libraries ──────────────────────────────────────────────────
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QuestionThreeSolutions:


    def __init__(self, initial_dataset_path,
                 imputed_training_dataset_path_year,
                 imputed_validation_dataset_path_year,
                 imputed_testing_dataset_path_year,
                 features,
                 response_variable,
                 prediction_size,
                 history_size):
        '''
        Docstring for __init__

        :param initial_dataset_path: the path there the initial dataset will be stored
        :param imputed_training_dataset_path_year: the path to the imputed training dataset for time series regression
        :param imputed_validation_dataset_path_year: the path to the imputed validation dataset for time series regression
        :param imputed_testing_dataset_path_year: the path to the imputed testing dataset for time series regression
        :param indicators: indicators that have to be downloaded from the world bank API
        :param features: numerical features that are used in the analysis process
        :param start_year: starting year of the data considered in the analysis, 1980 by default
        :param end_year: last year of the data considered in the analysis, 2023 by default
        '''

        self.initial_dataset_path = initial_dataset_path
        self.imputed_training_dataset_path_year = imputed_training_dataset_path_year
        self.imputed_validation_dataset_path_year = imputed_validation_dataset_path_year
        self.imputed_testing_dataset_path_year = imputed_testing_dataset_path_year
        self.features = features
        self.response_variable= response_variable
        self.prediction_size = prediction_size
        self.history_size=history_size

    def solve_question(self):

        '''
        Docstring for solve_question

        runner for question one
        please uncomment the function you want to be runned

        THIS FUNCTION RETURN THE DATASET CLASS THAT CONTAIN THE SEQUENCES FOR Q3
        '''

        # ──────────────────────── Create instance to store sequences for time series models ───────────────────────────────────────────────────
        ts = QuestionThreeSolutions.create_time_series(imputed_training_dataset_path=self.imputed_training_dataset_path_year, 
                                                    imputed_validation_dataset_path=self.imputed_validation_dataset_path_year,
                                                    imputed_testing_dataset_path=self.imputed_testing_dataset_path_year, 
                                                    features=self.features, 
                                                    response_variable=self.response_variable, 
                                                    prediction_size=self.prediction_size, 
                                                    history_size=self.history_size)
        
        # ──────────────────────── Function to train test and validate LSTM model ───────────────────────────────────────────────────
        set_seed(42)
        QuestionThreeSolutions.fit_LSTM_model(ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size)

        # ──────────────────────── Function to train test and validate CNN_LSTM model ───────────────────────────────────────────────────
        set_seed(42)
        QuestionThreeSolutions.fit_CNN_LSTM_model(ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size)

        # ──────────────────────── Function to train test and validate Transformer model ───────────────────────────────────────────────────
        set_seed(42)
        QuestionThreeSolutions.fit_Transformer_model(ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size)
        

    @staticmethod
    def create_time_series(imputed_training_dataset_path, 
                        imputed_validation_dataset_path,
                        imputed_testing_dataset_path, 
                        features, 
                        response_variable, 
                        prediction_size, 
                        history_size):
        
        ts = TimeSeries(imputed_training_dataset_path=imputed_training_dataset_path,
                        imputed_validation_dataset_path=imputed_validation_dataset_path,
                        imputed_testing_dataset_path=imputed_testing_dataset_path, 
                        features=features, response_variable=response_variable, 
                        test_size=prediction_size, window_length=history_size)
        
        return ts

    @staticmethod
    def fit_LSTM_model(ts, number_of_features, prediction_size):
            
        # ──────────────────────── Grid search over set of selected parameters ──────────────────────────────────────────────────
        '''grid_searcher = GridSearchTimeSeries( model_class = Pytorch_LSTM,
                                            model_fixed_params = {"num_features": number_of_features,
                                                                    "hidden_size":256,
                                                                    "num_layers":2,
                                                                    "drop_out": 0.1,
                                                                    "output_size": prediction_size},
                                            training_fixed_params = {                 
                                                "batch_size":    32,
                                                "learning_rate": 0.01,
                                                "num_epochs":    20,
                                            },
                                            ts                  = ts,
                                            training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                            optimizer           = optim.Adam,
                                            criterion           = nn.MSELoss(),
                                        )
        
        results = grid_searcher.run_grid_search(param_grid={
                    "hidden_size": [24, 48, 64, 80, 128, 256],
                    "num_layers":    [1, 2, 4],     
                    "drop_out":   [0.1, 0.3],
                    "num_epochs": [5, 10, 20]       
                }, top_n = 30)'''
        
        # ──────────────────────── Cross-validaiton over the batch_size ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Pytorch_LSTM,
                                                        model_fixed_params = {"num_features": number_of_features, 
                                                                            "output_size": prediction_size,
                                                                            "num_layers": 2,
                                                                            "drop_out": 0.1,
                                                                            "hidden_size": 256},
                                                        training_fixed_params = {                  
                                                            "batch_size":    48,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":    20,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"batch_size":[10,24, 32, 48, 64, 100, 128, 256]})'''
        # ──────────────────────── Cross-validaiton over the num_epochs ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Pytorch_LSTM,
                                                        model_fixed_params = {"num_features": number_of_features, 
                                                                            "output_size": prediction_size,
                                                                            "num_layers": 2,
                                                                            "drop_out": 0.1,
                                                                            "hidden_size": 256},
                                                        training_fixed_params = {                  
                                                            "batch_size":    64,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":    20,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"num_epochs":[5, 20, 40, 100, 200, 400]})'''
        
        # ──────────────────────── Cross-validaiton over the learning_rate  ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Pytorch_LSTM,
                                                        model_fixed_params = {"num_features": number_of_features, 
                                                                            "output_size": prediction_size,
                                                                            "num_layers": 2,
                                                                            "drop_out": 0.1,
                                                                            "hidden_size": 256},
                                                        training_fixed_params = {                  
                                                            "batch_size":    64,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":    200,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"learning_rate":[0.0001, 0.001, 0.005, 0.01, 0.1]})'''


        # ──────────────────────── Train model best hyperparameter set, plot the results and print out loss metrics ──────────────────────────────────────────────────
        set_seed(42)
        model = Pytorch_LSTM(hidden_size=256, 
                                num_layers=2, num_features=number_of_features, 
                                output_size=prediction_size, drop_out=0.1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        model.fit(train_data=ts.torch_training_time_series, num_epoches=200, batch_size=64, optimizer=optimizer, criterion=criterion)
        MSE_value, MAE_value = model.test_data(ts.torch_testing_time_series, 64)
        print("#" * 54)
        print(f"Overall testing dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        testing_single_country(ts = ts, model=model, country_name="United States", filename=(f'/Users/arina/Documents/results/LSTM/LSTM_USA'))
        testing_single_country(ts = ts, model=model, country_name="Russian Federation", filename=(f'/Users/arina/Documents/results/LSTM/LSTM_RF'))
        testing_single_country(ts = ts, model=model, country_name="China", filename=(f'/Users/arina/Documents/results/LSTM/LSTM_CH'))
        testing_single_country(ts = ts, model=model, country_name="Brazil", filename=(f'/Users/arina/Documents/results/LSTM/LSTM_BR'))

    @staticmethod
    def fit_CNN_LSTM_model(ts, number_of_features, prediction_size):

        # ──────────────────────── Grid search over set of selected parameters ──────────────────────────────────────────────────
        '''cnn_specs = [[{"filters_num": 128,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "max",
                                "pool_size": 3}, {"filters_num": 64,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "adaptive",
                                "pool_size": 10}],[{"filters_num": 64,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "max",
                                "pool_size": 3}, {"filters_num": 32,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "adaptive",
                                "pool_size": 10}], [{"filters_num": 128,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "max",
                                "pool_size": 3}, {"filters_num": 64,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "max",
                                "pool_size": 3}, {"filters_num": 32,
                                "kernel_size": 3,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "adaptive",
                                "pool_size": 10}], [{"filters_num": 64,
                                "kernel_size": 4,
                                "padding_size": 0,
                                "stride_size": 1,
                                "pool_type": "adaptive",
                                "pool_size": 10}],[{"filters_num": 32,
                                "kernel_size": 4,
                                "padding_size": 1,
                                "stride_size": 1,
                                "pool_type": "max",
                                "pool_size": 4}]]
        grid_searcher = GridSearchTimeSeries( model_class = CNN_LSTM,
                                                model_fixed_params = {"cnn_layers" : [{"filters_num": 128,
                                                "kernel_size": 3,
                                                "padding_size": 1,
                                                "stride_size": 1,
                                                "pool_type": "max",
                                                "pool_size": 2}, {"filters_num": 64,
                                                "kernel_size": 3,
                                                "padding_size": 1,
                                                "stride_size": 1,
                                                "pool_type": "adaptive",
                                                "pool_size": 10}], 
                                                "num_features": number_of_features,
                                                "hidden_size":128,
                                                "num_layers":1,
                                                "drop_out": 0.3,
                                                "output_size": prediction_size},
                                                training_fixed_params = {                  
                                                    "batch_size":    36,
                                                    "learning_rate": 0.01,
                                                    "num_epochs":    5,
                                                },
                                                ts                  = ts,
                                                training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                optimizer           = optim.Adam,
                                                criterion           = nn.MSELoss(),
                                                )
        
        results = grid_searcher.run_grid_search(param_grid={
                    "hidden_size": [24, 48, 64, 80, 128, 256],
                    "num_layers":    [1, 2, 4],     
                    "drop_out":   [0.1, 0.3],
                    "num_epochs": [5, 10, 20],  
                    "cnn_layers": cnn_specs}, top_n = 30)'''
        
        # ──────────────────────── Cross-validaiton over the batch_size ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = CNN_LSTM,
                                                model_fixed_params = {"cnn_layers" : [{'filters_num': 64, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'max', 'pool_size': 3}, 
                                                {'filters_num': 32, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'adaptive', 'pool_size': 10}], 
                                                "num_features": number_of_features,
                                                "hidden_size":24,
                                                "num_layers":1,
                                                "drop_out": 0.3,
                                                "output_size": prediction_size},
                                                training_fixed_params = {                  
                                                    "batch_size":    80,
                                                    "learning_rate": 0.01,
                                                    "num_epochs":    20,
                                                },
                                                ts                  = ts,
                                                training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                optimizer           = optim.Adam,
                                                criterion           = nn.MSELoss(),
                                                )
        cross_validator.run_cross_validation({"batch_size":[10,24, 32, 48, 64, 100, 128, 256]})'''
        

        # ──────────────────────── Cross-validaiton over the hidden_size ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = CNN_LSTM,
                                                model_fixed_params = {"cnn_layers" : [{'filters_num': 64, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'max', 'pool_size': 3}, 
                                                {'filters_num': 32, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'adaptive', 'pool_size': 10}], 
                                                "num_features": number_of_features,
                                                "hidden_size":24,
                                                "num_layers":1,
                                                "drop_out": 0.3,
                                                "output_size": prediction_size},
                                                training_fixed_params = {                  
                                                    "batch_size":    48,
                                                    "learning_rate": 0.01,
                                                    "num_epochs":    24,
                                                },
                                                ts                  = ts,
                                                training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                optimizer           = optim.Adam,
                                                criterion           =
                                                nn.MSELoss(),
                                                )
        cross_validator.run_cross_validation({"hidden_size":[12, 24, 48, 64, 80, 128, 200, 256]})'''

        # ──────────────────────── Cross-validaiton over the num_epochs ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = CNN_LSTM,
                                                model_fixed_params = {"cnn_layers" : [{'filters_num': 64, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'max', 'pool_size': 3}, 
                                                {'filters_num': 32, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'adaptive', 'pool_size': 10}], 
                                                "num_features": number_of_features,
                                                "hidden_size":128,
                                                "num_layers":1,
                                                "drop_out": 0.3,
                                                "output_size": prediction_size},
                                                training_fixed_params = {                  
                                                    "batch_size":    48,
                                                    "learning_rate": 0.01,
                                                    "num_epochs":    20,
                                                },
                                                ts                  = ts,
                                                training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                optimizer           = optim.Adam,
                                                criterion           = nn.MSELoss(),
                                                )
        cross_validator.run_cross_validation({"num_epochs":[5, 20, 40, 100, 200, 400]})'''


        # ──────────────────────── Cross-validaiton over the learning rate ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = CNN_LSTM,
                                                model_fixed_params = {"cnn_layers" : [{'filters_num': 64, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'max', 'pool_size': 3}, 
                                                {'filters_num': 32, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'adaptive', 'pool_size': 10}], 
                                                "num_features": number_of_features,
                                                "hidden_size":128,
                                                "num_layers":1,
                                                "drop_out": 0.3,
                                                "output_size": prediction_size},
                                                training_fixed_params = {                  
                                                    "batch_size":    48,
                                                    "learning_rate": 0.01,
                                                    "num_epochs":    100,
                                                },
                                                ts                  = ts,
                                                training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                optimizer           = optim.Adam,
                                                criterion           = nn.MSELoss(),
                                                )
        cross_validator.run_cross_validation({"learning_rate":[0.0001, 0.001, 0.005, 0.01, 0.1]})'''
        
        # ──────────────────────── Train model best hyperparameter set, plot the results and print out loss metrics ──────────────────────────────────────────────────
        set_seed(42)
        model = CNN_LSTM(cnn_layers=[{'filters_num': 64, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'max', 'pool_size': 3}, 
                                                {'filters_num': 32, 'kernel_size': 3, 'padding_size': 1, 
                                                'stride_size': 1, 'pool_type': 'adaptive', 'pool_size': 10}], 
                                                hidden_size=128, num_layers=1, num_features=number_of_features, 
                                                output_size=prediction_size, drop_out=0.3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        model.fit(train_data=ts.torch_training_time_series, num_epoches=100, batch_size=48, optimizer=optimizer, criterion=criterion)
        MSE_value, MAE_value = model.test_data(ts.torch_testing_time_series, 48)
        print("#" * 54)
        print(f"Overall testing dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        testing_single_country(ts = ts, model=model, country_name="United States", filename=(f'/Users/arina/Documents/results/CNN_LSTM/CNN_LSTM_USA'))
        testing_single_country(ts = ts, model=model, country_name="Russian Federation", filename=(f'/Users/arina/Documents/results/CNN_LSTM/CNN_LSTM_RF'))
        testing_single_country(ts = ts, model=model, country_name="China", filename=(f'/Users/arina/Documents/results/CNN_LSTM/CNN_LSTM_CH'))
        testing_single_country(ts = ts, model=model, country_name="Brazil", filename=(f'/Users/arina/Documents/results/CNN_LSTM/CNN_LSTM_BR'))

    @staticmethod
    def fit_Transformer_model(ts, number_of_features, prediction_size):

        # ──────────────────────── Grid search over set of selected parameters ──────────────────────────────────────────────────
        '''grid_searcher = GridSearchTimeSeries( model_class = Transformer,
                                            model_fixed_params = {"num_features":number_of_features,
                                                                "d_model":128,
                                                                "nhead":4,
                                                                "dim_feedforward": 64,
                                                                "dropout":0.2,
                                                                "num_layers":2,
                                                                "output_size": prediction_size},
                                            training_fixed_params = {                  
                                                "batch_size":    32,
                                                "learning_rate": 0.01,
                                                "num_epochs":    5,
                                            },
                                            ts                  = ts,
                                            training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                            optimizer           = optim.Adam,
                                            criterion           = nn.MSELoss(),
                                        )
        
        results = grid_searcher.run_grid_search(param_grid={
                    "d_model": [24, 32, 48, 64, 128],
                    "nhead": [2, 4, 8],
                    "dim_feedforward": [24, 32, 48, 64, 128],
                    "num_layers": [2, 4, 8],
                    "num_epochs": [5, 10, 20]}, top_n = 30)'''
        
        # ──────────────────────── Cross-validaiton over the batch size ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Transformer,
                                                        model_fixed_params = {"num_features":number_of_features,
                                                                "d_model":24,
                                                                "nhead":4,
                                                                "dim_feedforward": 24,
                                                                "dropout":0.2,
                                                                "num_layers":2,
                                                                "output_size": prediction_size},
                                                        training_fixed_params = {                  
                                                            "batch_size":    32,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":   20,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"batch_size":[10,24, 32, 48, 64, 100, 128, 256]})'''

        # ──────────────────────── Cross-validaiton over the epoch size ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Transformer,
                                                        model_fixed_params = {"num_features":number_of_features,
                                                                "d_model":24,
                                                                "nhead":4,
                                                                "dim_feedforward": 24,
                                                                "dropout":0.2,
                                                                "num_layers":2,
                                                                "output_size": prediction_size},
                                                        training_fixed_params = {                  
                                                            "batch_size":    64,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":   20,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"num_epochs":[5, 20, 40, 100, 200, 400]})'''

        # ──────────────────────── Cross-validaiton over the learning rate ──────────────────────────────────────────────────
        '''cross_validator = TimeSeriesCrossValidation( model_class = Transformer,
                                                        model_fixed_params = {"num_features":number_of_features,
                                                                "d_model":24,
                                                                "nhead":4,
                                                                "dim_feedforward": 24,
                                                                "dropout":0.2,
                                                                "num_layers":2,
                                                                "output_size": prediction_size},
                                                        training_fixed_params = {                  
                                                            "batch_size":    64,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":   5,
                                                        },
                                                        ts                  = ts,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                        criterion           = nn.MSELoss(),
                                                    )
        cross_validator.run_cross_validation({"learning_rate":[0.0001, 0.001, 0.005, 0.01, 0.1]})'''

        # ──────────────────────── Train model best hyperparameter set, plot the results and print out loss metrics ──────────────────────────────────────────────────
        set_seed(42)
        model = Transformer(num_features=number_of_features, 
                            d_model=24, 
                            nhead=4, 
                            dim_feedforward=24, 
                            dropout=0.2, 
                            num_layers=2, 
                            output_size=prediction_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        model.fit(ts.torch_training_time_series, 5, 64, optimizer, criterion)

        MSE_value, MAE_value = model.test_data(ts.torch_testing_time_series, 64)
        print("#" * 54)
        print(f"Overall testing dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        testing_single_country(ts = ts, model=model, country_name="United States", filename=(f'/Users/arina/Documents/results/transformer/TRF_USA'))
        testing_single_country(ts = ts, model=model, country_name="Russian Federation", filename=(f'/Users/arina/Documents/results/transformer/TRF_RF'))
        testing_single_country(ts = ts, model=model, country_name="China", filename=(f'/Users/arina/Documents/results/transformer/TRF_CH'))
        testing_single_country(ts = ts, model=model, country_name="Brazil", filename=(f'/Users/arina/Documents/results/transformer/TRF_BR'))

def testing_single_country(ts, model, country_name, filename):

    test_point, test_scaler = ts.get_single_testing(country_name)
    predictions, actuals = model.test_single_point(test_point, test_scaler)
    
    mae=MAE(actuals, predictions)
    mape=MAPE(actuals, predictions)
    mse=MSE(actuals, predictions)
    print("#" * 54)
    print(f"MAE value for {country_name}: {mae}")
    print(f"MAPE value for {country_name}: {mape}")
    print(f"MSE value for {country_name}: {mse}")
    print(f"Values Predicted by model for {country_name}: {predictions}")
    print(f"Actual value for {country_name}: {actuals}")
    print("#" * 54)

    plt.figure(figsize=(8,6))
    plt.plot(actuals, color="red", label='Actual Data')
    plt.plot(predictions, color = "blue", label = 'Predicted Data')
    plt.xlabel('Years')
    plt.ylabel('GDP')
    plt.title('Actual vs. Predicted Observations')
    plt.legend()
    if filename:
        plt.savefig(filename, dpi = 600)
        plt.close()
    else:
        plt.show()

def MAE(actuals, predicted):
    return np.mean(np.abs(actuals-predicted))

def MAPE(actuals, predicted):
    return np.mean(np.abs((actuals-predicted)/actuals))*100

def MSE(actuals, predicted):
    return np.mean((actuals-predicted)**2)