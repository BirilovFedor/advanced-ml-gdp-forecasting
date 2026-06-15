# ──────────────────────── import from the project ──────────────────────────────────────────────────
from DataStorage.TimeSeries import TimeSeries
from DataStorage.FlattenedSequence import FlattenedSequence
from Models.Variational_Autoencoder import VariationalAutoencoder
from Models.PytorchLSTM import Pytorch_LSTM
from Models.CNN_LSTM import CNN_LSTM
from Models.Transformer import Transformer

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


class QuestionFiveSolutions:


    def __init__(self,
                 imputed_training_dataset_path_year,
                 imputed_validation_dataset_path_year,
                 imputed_testing_dataset_path_year,
                 features,
                 response_variable,
                 prediction_size,
                 history_size):
        '''
        Docstring for __init__

        :param imputed_training_dataset_path_year: the path to the imputed training dataset for time series regression
        :param imputed_validation_dataset_path_year: the path to the imputed validation dataset for time series regression
        :param imputed_testing_dataset_path_year: the path to the imputed testing dataset for time series regression
        :param indicators: indicators that have to be downloaded from the world bank API
        :param features: numerical features that are used in the analysis process
        :param start_year: starting year of the data considered in the analysis, 1980 by default
        :param end_year: last year of the data considered in the analysis, 2023 by default
        '''

        self.imputed_training_dataset_path_year = imputed_training_dataset_path_year
        self.imputed_validation_dataset_path_year = imputed_validation_dataset_path_year
        self.imputed_testing_dataset_path_year = imputed_testing_dataset_path_year
        self.features = features
        self.response_variable= response_variable
        self.prediction_size = prediction_size
        self.history_size=history_size
        self.variational_autoencoder_input_size = self.history_size*len(self.features)+self.prediction_size

    def solve_question(self):

        '''
        Docstring for solve_question

        runner for question one
        please uncomment the function you want to be runned

        THIS FUNCTION RETURN THE DATASET CLASS THAT CONTAIN THE SEQUENCES FOR Q5
        '''
        
        # ──────────────────────── Create instance to store sequences for time series models ───────────────────────────────────────────────────
        ts = QuestionFiveSolutions.create_time_series(imputed_training_dataset_path=self.imputed_training_dataset_path_year, 
                                                    imputed_validation_dataset_path=self.imputed_validation_dataset_path_year,
                                                    imputed_testing_dataset_path=self.imputed_testing_dataset_path_year, 
                                                    features=self.features, 
                                                    response_variable=self.response_variable, 
                                                    prediction_size=self.prediction_size, 
                                                    history_size=self.history_size)

        # ──────────────────────── Create flattened sequences of concatenated input output pairs to feed Autoencoder ───────────────────────────────────────────────────
        fl = QuestionFiveSolutions.create_flattened_sequence(ts=ts)
        
        # ──────────────────────── Train Autoencoder ───────────────────────────────────────────────────
        model = QuestionFiveSolutions.fit_Autoencoder(fl, self.variational_autoencoder_input_size, 0.2)

        # ──────────────────────── Sample from the latent space ───────────────────────────────────────────────────
        samples = QuestionFiveSolutions.sample_from_latent_space(model, 2000)

        ts.add_simulated_data(samples)

        # ──────────────────────── Transformed augmented vs normal ───────────────────────────────────────────────────
        QuestionFiveSolutions.plot_normal_vs_augmented(function=QuestionFiveSolutions.fit_Transformer_model, ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size, batch_size=5, model_name="Transformer")
        
        # ──────────────────────── CNN-LSTM augmented vs normal ───────────────────────────────────────────────────
        QuestionFiveSolutions.plot_normal_vs_augmented(function=QuestionFiveSolutions.fit_CNN_LSTM_model, ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size, batch_size=48, model_name="CNNLSTM")
        
        # ──────────────────────── LSTM augmented vs normal ───────────────────────────────────────────────────
        QuestionFiveSolutions.plot_normal_vs_augmented(function=QuestionFiveSolutions.fit_LSTM_model, ts=ts, number_of_features=len(self.features), prediction_size=self.prediction_size, batch_size=64, model_name="LSTM")


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
    def create_flattened_sequence(ts):
            
        fl = FlattenedSequence(ts.training_time_series, 
                                   ts.validation_time_series, 
                                   ts.testing_time_series)
            
        return fl
    
    @staticmethod
    def fit_Autoencoder(fl, input_dim, beta, file_name=''):
        #f'/Users/arina/Documents/results/Autoencoder/BestModelTraining'
        # ──────────────────────── Fit Variational Aut ───────────────────────────────────────────────────
        # Identify the specifications for Encoder 
        encoder_specs = [{'linear_ouput': 64, 'is_relu': True}]
        # Identify the specifications for Decoder 
        decoder_specs = [{'linear_ouput': 64, 'is_relu': True}]

        set_seed(42)
        model = VariationalAutoencoder(input_dim = input_dim, latent_dim=64, encoder_specifications=encoder_specs,decoder_specifications= decoder_specs)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        total_losses, recon_losses, kl_losses = model.fit(fl.torch_flattened_training_data, 200, 48, optimizer)
        return model

    @staticmethod
    def fit_LSTM_model(ts, training_data, testing_data, number_of_features, prediction_size):
          
        # ──────────────────────── Train model best hyperparameter set, plot the results and print out loss metrics ──────────────────────────────────────────────────
        set_seed(42)
        model = Pytorch_LSTM(hidden_size=256, 
                                num_layers=2, num_features=number_of_features, 
                                output_size=prediction_size, drop_out=0.1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        model.fit(train_data=training_data, num_epoches=200, batch_size=64, optimizer=optimizer, criterion=criterion)
        return model
    
    @staticmethod
    def fit_CNN_LSTM_model(ts, training_data, testing_data, number_of_features, prediction_size):

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
        model.fit(train_data=training_data, num_epoches=100, batch_size=48, optimizer=optimizer, criterion=criterion)
        return model

    @staticmethod
    def fit_Transformer_model(ts, training_data, testing_data, number_of_features, prediction_size):

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
        model.fit(training_data, 5, 64, optimizer, criterion)

        return model

    @staticmethod
    def sample_from_latent_space(model, num_samples):
        set_seed(42)
        return model.generate_syntetic_data(num_samples)

    @staticmethod
    def plot_normal_vs_augmented(function, ts, number_of_features, prediction_size, batch_size, model_name):

        model = function(ts=ts,
                                                    training_data=ts.torch_training_time_series,
                                                    testing_data=ts.torch_testing_time_series,
                                                    number_of_features=number_of_features,
                                                    prediction_size=prediction_size)
        
        model_augmented = function(ts=ts,
                                                    training_data=ts.torch_sim_training_time_series,
                                                    testing_data=ts.torch_testing_time_series,
                                                    number_of_features=number_of_features,
                                                    prediction_size=prediction_size)
        print(f"RESULT FOR {function}")
        print(120*'#')
        MSE_value, MAE_value = model.test_data(ts.torch_testing_time_series, batch_size)
        print("#" * 54)
        print(f"Overall testing dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        MSE_value, MAE_value = model_augmented.test_data(ts.torch_testing_time_series, batch_size)
        print("#" * 54)
        print(f"Overall testing dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        MSE_value, MAE_value = model.test_data(ts.torch_validation_time_series, batch_size)
        print("#" * 54)
        print(f"Overall validation dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)
        MSE_value, MAE_value = model_augmented.test_data(ts.torch_validation_time_series, batch_size)
        print("#" * 54)
        print(f"Overall validation dataset results: MSE = {MSE_value}, MAE = {MAE_value}")
        print("#" * 54)

        QuestionFiveSolutions.plot_normal_vs_augmented_single_country(ts=ts, model=model, model_augmented=model_augmented, country_name="United States", file_name=f'/Users/arina/Documents/results/Augmentation/{model_name}_US')
        QuestionFiveSolutions.plot_normal_vs_augmented_single_country(ts=ts, model=model, model_augmented=model_augmented, country_name="Russian Federation", file_name=f'/Users/arina/Documents/results/Augmentation/{model_name}_RF')
        QuestionFiveSolutions.plot_normal_vs_augmented_single_country(ts=ts, model=model, model_augmented=model_augmented, country_name="China", file_name=f'/Users/arina/Documents/results/Augmentation/{model_name}_CH')
        QuestionFiveSolutions.plot_normal_vs_augmented_single_country(ts=ts, model=model, model_augmented=model_augmented, country_name="Brazil", file_name=f'/Users/arina/Documents/results/Augmentation/{model_name}_BR')

    @staticmethod
    def plot_normal_vs_augmented_single_country(ts, model, model_augmented, country_name, file_name):

        test_point, test_scaler = ts.get_single_testing(country_name)
        predictions, actuals = model.test_single_point(test_point, test_scaler)
        predictions_augmented, actuals_augmented = model_augmented.test_single_point(test_point, test_scaler)

        mae=MAE(actuals_augmented, predictions_augmented)
        mape=MAPE(actuals_augmented, predictions_augmented)
        mse=MSE(actuals_augmented, predictions_augmented)

        print(f"#" * 54)
        print(f"MAE value for {country_name}: {mae}")
        print(f"MAPE value for {country_name}: {mape}")
        print(f"MSE value for {country_name}: {mse}")
        print(f"Values Predicted by model for {country_name}: {predictions}")
        print(f"Actual value for {country_name}: {actuals}")
        print("#" * 54)

        plt.figure(figsize=(8,6))
        plt.plot(actuals, color="red", label='Actual Data')
        plt.plot(predictions, color = "blue", label = 'Predicted Data')
        plt.plot(predictions_augmented, color = "green", label = 'Predicted Data with Augmentation')
        plt.xlabel('Years')
        plt.ylabel('GDP')
        plt.title('Actual vs. Predicted Observations')
        plt.legend()
        if file_name:
            plt.savefig(file_name, dpi = 600)
            plt.close()
        else:
            plt.show()

def MAE(actuals, predicted):
    return np.mean(np.abs(actuals-predicted))

def MAPE(actuals, predicted):
    return np.mean(np.abs((actuals-predicted)/actuals))*100

def MSE(actuals, predicted):
    return np.mean((actuals-predicted)**2)