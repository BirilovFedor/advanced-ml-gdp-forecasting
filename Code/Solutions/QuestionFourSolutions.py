# ──────────────────────── import from the project ──────────────────────────────────────────────────
from DataStorage.TimeSeries import TimeSeries
from DataStorage.FlattenedSequence import FlattenedSequence
from Models.Variational_Autoencoder import VariationalAutoencoder
from Validation.GridSearchAutoencoder import GridSearchAutoencoder
from Validation.CrossValidationAutoencoder import CrossValidationAutoencoder

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


class QuestionFourSolutions:


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

        THIS FUNCTION RETURN THE DATASET CLASS THAT CONTAIN THE SEQUENCES FOR Q4
        '''
        
        # ──────────────────────── Create instance to store sequences for time series models ───────────────────────────────────────────────────
        ts = QuestionFourSolutions.create_time_series(imputed_training_dataset_path=self.imputed_training_dataset_path_year, 
                                                    imputed_validation_dataset_path=self.imputed_validation_dataset_path_year,
                                                    imputed_testing_dataset_path=self.imputed_testing_dataset_path_year, 
                                                    features=self.features, 
                                                    response_variable=self.response_variable, 
                                                    prediction_size=self.prediction_size, 
                                                    history_size=self.history_size)

        # ──────────────────────── Create flattened sequences of concatenated input output pairs to feed Autoencoder ───────────────────────────────────────────────────
        fl = QuestionFourSolutions.create_flattened_sequence(ts=ts)

        # ──────────────────────── Grid Search over paramerters ───────────────────────────────────────────────────
        #QuestionFourSolutions.grid_search(fl=fl, input_dim=self.variational_autoencoder_input_size)
        
        # ──────────────────────── Cross-Validation over paramerters ───────────────────────────────────────────────────
        #QuestionFourSolutions.cross_validation(fl=fl, input_dim=self.variational_autoencoder_input_size)

        # ──────────────────────── Fit best Autoencoder ───────────────────────────────────────────────────
        #QuestionFourSolutions.fit_Autoencoder(fl, self.variational_autoencoder_input_size, 0.2)

        # ──────────────────────── Plot the latent representation of the data ───────────────────────────────────────────────────

        '''encoder_specs = [{'linear_ouput': 64, 'is_relu': True}]
        decoder_specs = [{'linear_ouput': 64, 'is_relu': True}]

        QuestionFourSolutions.visualize_reduced_latent_representation(fl=fl, input_dim= self.variational_autoencoder_input_size, 
                                                latent_dim=64, encoder_specifications=encoder_specs,
                                                decoder_specifications=decoder_specs, learning_rate=0.005, num_epoches = 200, batch_size = 48)'''

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
        plt.figure(figsize=(8,6))
        plt.plot(total_losses, color = "red", label = "Total Loss")
        plt.plot(recon_losses, color = "blue", label = "Reconstruction Loss")
        plt.plot(kl_losses, color = "green", label = "KL Convergence Loss")
        plt.title('Variational Autoencoder Performence in Different Loss Metrics')
        plt.xlabel("Batch")
        plt.ylabel('Loss Criteria')
        plt.legend()
        if file_name:
            plt.savefig(file_name, dpi=600)
            plt.close()
        else:
            plt.show()
        print(f"Reconstruction Loss {recon_losses[-1]}")
        print(f"KL Convergence Loss: {kl_losses[-1]}")
        print(f"ELBO: {-(recon_losses[-1] + beta*kl_losses[-1])}")

    @staticmethod
    def cross_validation(fl, input_dim):

        # ──────────────────────── Cross-Validation over the batch_size ───────────────────────────────────────────────────
        '''# Identify the specifications for Encoder 
        encoder_specs = [{'linear_ouput': 64, 'is_relu': True}]
        # Identify the specifications for Decoder 
        decoder_specs = [{'linear_ouput': 64, 'is_relu': True}]

        cross_validator = CrossValidationAutoencoder( model_class = VariationalAutoencoder,
                                                        model_fixed_params = {"input_dim":input_dim,
                                                                "latent_dim":64,
                                                                "encoder_specifications":encoder_specs,
                                                                "decoder_specifications": decoder_specs
                                                                },
                                                        training_fixed_params = {                  
                                                            "batch_size":    20,
                                                            "learning_rate": 0.0001,
                                                            "num_epochs":    100,
                                                        },
                                                        ts                  = fl,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                    )

        
        cross_validator.run_cross_validation({"batch_size":[6, 12, 24, 36, 48, 64, 96, 128]})'''

        # ──────────────────────── Cross-Validation over the num_epochs ───────────────────────────────────────────────────
        '''# Identify the specifications for Encoder 
        encoder_specs = [{'linear_ouput': 64, 'is_relu': True}]
        # Identify the specifications for Decoder 
        decoder_specs = [{'linear_ouput': 64, 'is_relu': True}]

        cross_validator = CrossValidationAutoencoder( model_class = VariationalAutoencoder,
                                                        model_fixed_params = {"input_dim":input_dim,
                                                                "latent_dim":64,
                                                                "encoder_specifications":encoder_specs,
                                                                "decoder_specifications": decoder_specs
                                                                },
                                                        training_fixed_params = {                  
                                                            "batch_size":    48,
                                                            "learning_rate": 0.0001,
                                                            "num_epochs":    100,
                                                        },
                                                        ts                  = fl,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                    )

        
        cross_validator.run_cross_validation({"num_epochs":[5, 10, 20, 40, 80, 100, 150, 200, 400]})'''

        # ──────────────────────── Cross-Validation over the learning_rate ───────────────────────────────────────────────────
        '''# Identify the specifications for Encoder 
        encoder_specs = [{'linear_ouput': 64, 'is_relu': True}]
        # Identify the specifications for Decoder 
        decoder_specs = [{'linear_ouput': 64, 'is_relu': True}]

        cross_validator = CrossValidationAutoencoder( model_class = VariationalAutoencoder,
                                                        model_fixed_params = {"input_dim":input_dim,
                                                                "latent_dim":64,
                                                                "encoder_specifications":encoder_specs,
                                                                "decoder_specifications": decoder_specs
                                                                },
                                                        training_fixed_params = {                  
                                                            "batch_size":    48,
                                                            "learning_rate": 0.0001,
                                                            "num_epochs":    200,
                                                        },
                                                        ts                  = fl,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                    )

        
        cross_validator.run_cross_validation({"learning_rate":[0.0001,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]})'''

    @staticmethod
    def grid_search(fl, input_dim):
        # Identify the specifications for Encoder 
        encoder_specs = [{"linear_ouput": 64,"is_relu":True}, {"linear_ouput": 32,"is_relu":False}]
        # Identify the specifications for Decoder 
        decoder_specs = [{"linear_ouput": 32,"is_relu":True}]

        grid_searcher = GridSearchAutoencoder( model_class = VariationalAutoencoder,
                                                        model_fixed_params = {"input_dim":input_dim,
                                                                "latent_dim":36,
                                                                "encoder_specifications":encoder_specs,
                                                                "decoder_specifications": decoder_specs
                                                                },
                                                        training_fixed_params = {                  
                                                            "batch_size":    20,
                                                            "learning_rate": 0.01,
                                                            "num_epochs":    5,
                                                        },
                                                        ts                  = fl,
                                                        training_param_keys = ["batch_size", "learning_rate", "num_epochs"],
                                                        optimizer           = optim.Adam,
                                                    )
        encoder_specifications =[ [{"linear_ouput": 64,"is_relu":True}, {"linear_ouput": 32,"is_relu":False}],  [{"linear_ouput": 32,"is_relu":True}, {"linear_ouput": 16,"is_relu":False}], [{"linear_ouput": 64,"is_relu":True}], [{"linear_ouput": 64,"is_relu":True}, {"linear_ouput": 32,"is_relu":True}], [{"linear_ouput": 128,"is_relu":True}, {"linear_ouput": 64,"is_relu":False}]] 
        decoder_specifications = [ [{"linear_ouput": 64,"is_relu":True}, {"linear_ouput": 32,"is_relu":False}],  [{"linear_ouput": 32,"is_relu":True}, {"linear_ouput": 16,"is_relu":False}], [{"linear_ouput": 64,"is_relu":True}], [{"linear_ouput": 64,"is_relu":True}, {"linear_ouput": 32,"is_relu":True}], [{"linear_ouput": 128,"is_relu":True}, {"linear_ouput": 64,"is_relu":False}]]
        results = grid_searcher.run_grid_search(param_grid={
                    "encoder_specifications": encoder_specifications,
                    "decoder_specifications": decoder_specifications,
                    "latent_dim":    [8, 16, 32, 64],
                    "num_epochs": [5, 10, 40, 100],
                    "learning_rate": [0.0001, 0.01]}, top_n = 30)
        
    @staticmethod
    def visualize_reduced_latent_representation(fl, input_dim, latent_dim, encoder_specifications, decoder_specifications, learning_rate, num_epoches, batch_size):
        model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim, encoder_specifications=encoder_specifications, decoder_specifications=decoder_specifications)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.fit(fl.torch_flattened_training_data, num_epoches, batch_size, optimizer)
        mu, logvar = model.get_vae_latent_means(fl.flattened_training_data)
        reduced_mu = model.reduce_mean_dim(mu, 3)
        plot_latent_means_3d(reduced_mu)


def plot_latent_means_3d(means, labels=None, label_names=None, title="3D Latent Representation", file_name = ''):
    #'/Users/arina/Documents/results/Autoencoder/LatentRepresentation'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")


    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))


    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    if labels is None:
        ax.scatter(
            means[:, 0], means[:, 1], means[:, 2],
            c="royalblue", s=28, alpha=0.55
        )
    else:
        unique_labels = np.unique(labels)
        cmap = plt.cm.Set2

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            color = cmap(i / max(1, len(unique_labels) - 1))

            if isinstance(label_names, dict):
                name = label_names.get(lbl, f"Cluster {lbl}")
            elif isinstance(label_names, list):
                name = label_names[i] if i < len(label_names) else f"Cluster {lbl}"
            else:
                name = f"Cluster {lbl}"

            ax.scatter(
                means[mask, 0], means[mask, 1], means[mask, 2],
                c=[color], s=35, alpha=0.65, label=name
            )

        ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9)


    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.set_xlabel("Dim 1", fontsize=11, labelpad=10)
    ax.set_ylabel("Dim 2", fontsize=11, labelpad=10)
    ax.set_zlabel("Dim 3", fontsize=11, labelpad=10)
    ax.view_init(elev=25, azim=235)

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=600)
        plt.close()
    else:
        plt.show()
    