import pandas as pd 
import numpy as np
from SOLUTIONS.Q4SOL import QuestionFour
from DataStorage.Dataset import Dataset
from DataStorage.TimeSeries import TimeSeries
from Models.Variational_Autoencoder import VariationalAutoencoder
from Validation.TimeSeriesCrossValidation import TimeSeriesCrossValidation
from DataStorage.HybridImputer import HybridImputer
from Models.MLP import MLP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Validation.CrossValidation import CrossValidation
from Models.LSTM import LSTM
from Models.PytorchLSTM import Pytorch_LSTM
from Models.CNN_LSTM import CNN_LSTM
from Models.Transformer import Transformer
from DataStorage.FlattenedSequence import FlattenedSequence
import os
pd.set_option("display.max_columns", None)
from SOLUTIONS.Q3SOL import QuestionThree

import torch.nn as nn
import torch.optim as optim







def main():
    # Show all rows
    pd.set_option('display.max_rows', None)

    # Show all columns
    pd.set_option('display.max_columns', None)

    # Show full column width
    pd.set_option('display.max_colwidth', None)
    
    
    QuestionFour()
    
    #QuestionThree()
    features = ["GDPpc_2017$","Population_total","Life_exectancy","Literacy_rate","Unemploymlent_rate","Energy_use","Fertility_rate","Poverty_ratio","Primary_school_enrolmet_rate","Exports_2017$"]
    
    features2 = ["GDPpc_2017$","Population_total","Life_exectancy","Gross_capital_formation","Unemploymlent_rate","Energy_use","Fertility_rate","Poverty_ratio","Primary_school_enrolmet_rate","Exports_2017$"]

    initial_dataset_path = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\world_bank_data_dev80-23++.csv"
    initial_dataset_path_2 = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\final_data.csv"
    imputed_training_dataset_path_country = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\imputed_training_dataset_country.csv"
    imputed_testing_dataset_path_country = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\imputed_testing_dataset_country.csv"

    imputed_training_dataset_path_year = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\imputed_training_dataset_year.csv"
    imputed_validation_dataset_path_year = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\imputed_validation_dataset_year.csv"
    imputed_testing_dataset_path_year = "C:\\Users\\biril\\Desktop\\AdvancedMLAssignment\\Documentation\\imputed_testing_dataset_year.csv"
    
    #imputation_year(features, initial_dataset_path, imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year)

    #fit_Transformer_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features)

    #fit_LSTM_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features)
    #fit_CNNLSTM_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features)

    '''cnnlstm_model = CNN_LSTM([{"filters_num": 36,
                              "kernel_size": 3,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "max",
                              "pool_size": 2}, {"filters_num": 64,
                              "kernel_size": 3,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "adaptive",
                              "pool_size": 10}], hidden_size=36, num_layers=2, num_features=10, output_size=5, drop_out=0.1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(cnnlstm_model.parameters()
                           , lr=0.01)
    cnnlstm_model.fit(ts.torch_training_time_series, 20, 20, optimizer, criterion)
    predictions, actuals = cnnlstm_model.predict(ts.torch_validation_time_series, 20, ts.scaler)
    print(predictions)
    print(actuals)'''

    #MLP_training(imputed_training_dataset_path_country, imputed_testing_dataset_path_country)



def plot_latent_means_3d(means, labels=None, label_names=None, title="3D Latent Means"):


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("#0d1117")

    if labels is None:
        ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                   color="cyan", s=30, alpha=0.7)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("plasma", len(unique_labels))

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl

            # color
            color = cmap(i)

            # label name handling
            if isinstance(label_names, dict):
                name = label_names.get(lbl, f"Cluster {lbl}")
            elif isinstance(label_names, list):
                name = label_names[i] if i < len(label_names) else f"Cluster {lbl}"
            else:
                name = f"Cluster {lbl}"

            ax.scatter(means[mask, 0], means[mask, 1], means[mask, 2],
                       color=color, s=40, alpha=0.7,
                       label=name, edgecolors="none")

        ax.legend(loc="upper left", framealpha=0.3, fontsize=9)

    # Styling
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dim 1", color="white")
    ax.set_ylabel("Dim 2", color="white")
    ax.set_zlabel("Dim 3", color="white")

    ax.tick_params(colors="white")

    # dark theme panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")

    plt.show()

def imputation_year(features, initial_dataset_path, imputed_training_dataset_path, imputed_validation_dataset_path, imputed_testing_dataset_path, validation_size, test_size):
    '''
    Docstring for imputation_year:
    :param features: features of the initial dataset
    :param initial_dataset_path: path to the dataset that have to be imputed
    :param imputed_training_dataset_path: path to store imputed training data
    :param imputed_validation_dataset_path: path to store imputed validation data
    :param imputed_testing_dataset_path:  path to store imputed testing data
    :param validation_size: number of years for validation subset
    :param test_size: number of years for testing subset

    This function is reading the initial fail from a csv fail, apply imputation, including linear iterpolation, forward and backward fill,
    and kNN imputer. The data is splitted in to training, validation split prior kNN imputer fitting to insure no data leakage. For this split
    the cubic_splin and linear interpolations are using only past information to ensure no data leakage 
    '''

    # ──────────────────────── Create the imputer instance ───────────────────────────────────────────────────
    imputer = HybridImputer(path=initial_dataset_path, features=features, split_criteria='year', year_split_sizes=[validation_size, test_size])
    # ──────────────────────── Apply imputation to the dataset ───────────────────────────────────────────────────
    imputer.imputation()

    # ──────────────────────── Save files  ───────────────────────────────────────────────────
    imputer.training_data.to_csv(imputed_training_dataset_path, index=False)
    print(imputer.training_data.isna().sum().sum())
    imputer.validation_data.to_csv(imputed_validation_dataset_path, index=False)
    imputer.testing_data.to_csv(imputed_testing_dataset_path, index=False)

    print(len(imputer.training_data))
    print(len(imputer.validation_data))
    print(len(imputer.testing_data))


def fit_Transformer_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features):

    ts = TimeSeries(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features, "GDPpc_2017$", 5, 10)

    model = Transformer(num_features=10, d_model=24, nhead=4, dim_feedforward=48, dropout=0.2, num_layers=2, output_size=5)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters()
                           , lr=0.1) 
    
    model.fit(ts.torch_training_time_series, 10, 20, optimizer, criterion)
    predictions, actuals = model.predict(ts.torch_validation_time_series, 20, ts.scaler)
    print(predictions)
    print(actuals)


def fit_LSTM_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features):

    ts = TimeSeries(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features, "GDPpc_2017$", 5, 10)

    lstmmodel = Pytorch_LSTM(hidden_size=36, num_layers=2, num_features=10, output_size=5, drop_out=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstmmodel.parameters()
                           , lr=0.1)
    lstmmodel.fit(ts.torch_training_time_series, 10, 20, optimizer, criterion)
    predictions, actuals = lstmmodel.predict(ts.torch_validation_time_series, 20)
    print(predictions)
    print(actuals)

    test_point, test_scaler = ts.get_single_testing("Russian Federation")
    predictions, actuals = lstmmodel.test_single_point(test_point, test_scaler)
    print(predictions)
    print(actuals)

def fit_CNNLSTM_and_plot_testing_vs_training(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features):
    
    cnn_specs = [{"filters_num": 128,
                              "kernel_size": 4,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "adaptive",
                              "pool_size":10}]
    
    ts = TimeSeries(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features, "GDPpc_2017$", 5, 10)

    model = CNN_LSTM(cnn_layers=cnn_specs, hidden_size=64, num_layers=1, num_features=10, output_size=5, drop_out=0.1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters()
                           , lr=0.01)
    model.fit(ts.torch_training_time_series, 40, 40, optimizer, criterion)
    predictions, actuals = model.predict(ts.torch_validation_time_series, 20)
    print(predictions)
    print(actuals)

    test_point, test_scaler = ts.get_single_testing("Russian Federation")
    predictions, actuals = model.test_single_point(test_point, test_scaler)
    print(predictions)
    print(actuals)

    '''ts = TimeSeries(imputed_training_dataset_path_year, imputed_validation_dataset_path_year, imputed_testing_dataset_path_year, features, "GDPpc_2017$", 5, 10)
    validator = TimeSeriesCrossValidator(ts)
    training_L1, validation_L1 = validator.cross_validation(20, [0.001, 0.01, 0.05, 0.1, 0.5], 20, "learning_rate")
    plt.plot([0.001, 0.01, 0.05, 0.1, 0.5], training_L1, color='blue', label='training L1')
    plt.plot([0.001, 0.01, 0.05, 0.1, 0.5], validation_L1, color='red', label='validation L1')
    plt.show()
    cnn_specs = [[{"filters_num": 36,
                              "kernel_size": 3,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "max",
                              "pool_size": 2}, {"filters_num": 64,
                              "kernel_size": 3,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "adaptive",
                              "pool_size": 10}], [{"filters_num": 36,
                              "kernel_size": 3,
                              "padding_size": 1,
                              "stride_size": 1,
                              "pool_type": "adaptive",
                              "pool_size": 10}]]
    validator.grid_search(batch_sizes=[5], learning_rates=[0.001], num_epoches=[1],
                    hidden_sizes=[36], num_layers=[1], cnn_specs=cnn_specs, optimizer=optim.Adam, criterion=nn.L1Loss())'''


def mem(initial_dataset_path, sep):

    data = read_csv(initial_dataset_path, sep=sep)
    print(data.head())

    
    train_df = pd.concat([
    group.sort_values("date").iloc[:-5]
    for country, group in data.groupby("country")
    ])

    test_df = pd.concat([
        group.sort_values("date").iloc[-5:]
        for country, group in data.groupby("country")
    ])
    print(test_df.head())



def read_csv(path, sep):
    # function to read the data from the csv file

    return pd.read_csv(path, sep=sep)

def imputation_country(features, initial_dataset_path, imputed_training_dataset_path, imputed_testing_dataset_path):

    imputer = HybridImputer(path=initial_dataset_path, features=features, split_criteria='country')
    imputer.imputation()

    imputer.training_data.to_csv(imputed_training_dataset_path, index=False)
    imputer.testing_data.to_csv(imputed_testing_dataset_path, index=False)



def MLP_training(imputed_training_dataset_path_country, imputed_testing_dataset_path_country):


    dataset = Dataset(imputed_training_dataset_path=imputed_training_dataset_path_country, imputed_testing_dataset_path=imputed_testing_dataset_path_country)

    dataset.get_embedding()

    X_training = np.vstack(dataset.training_embedding['embedding'].to_list())
    X_testing = np.vstack(dataset.testing_embedding['embedding'].to_list())

    y_training = np.array(dataset.training_embedding['label'].to_list())
    y_testing = np.array(dataset.testing_embedding['label'].to_list())

    scaler = StandardScaler()
    scaled_X_training = scaler.fit_transform(X_training)
    scaled_X_testing = scaler.transform(X_testing)

    mlp_model = MLP([X_training.shape[1], 128, 64, 4])

    #fit_MLP_and_plot_testing_vs_training([X_training.shape[1], 256, 4], scaled_X_training, y_training, scaled_X_testing, y_testing, batch_size=10, learning_rate=0.001)
    #fit_MLP_and_plot_testing_vs_training([X_training.shape[1], 128, 64, 4], scaled_X_training, y_training, scaled_X_testing, y_testing, batch_size=10, learning_rate=0.001)

    #validator = CrossValidation(X_training, y_training)

    
    #print(validator.cross_validation(5, mlp_model, 10, 0.01, 200))

    apply_cross_validation_and_plot([X_training.shape[1], 128, 64, 4],scaled_X_training, scaled_X_testing, y_testing, X_training, y_training, 5, [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 100, "learning_rate")
    #apply_cross_validation_and_plot([X_training.shape[1], 128, 64, 4], scaled_X_training, scaled_X_testing, y_testing, X_training, y_training, [1, 2, 5, 10, 15, 20, 50, 100], 0.001, 100, "batch_size")
    #apply_cross_validation_and_plot([X_training.shape[1], 128, 64, 4], scaled_X_training, scaled_X_testing, y_testing, X_training, y_training, 5, 0.001, [20, 40, 60, 80, 100, 200, 300, 400, 600, 800], "num_epoches")


def apply_cross_validation_and_plot(layer_sizes, scaled_X_training, scaled_X_testing, y_testing, X_training, y_training, batch_size, learning_rate, num_epoches, indicator):

    params = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epoches": num_epoches
    }

    validator = CrossValidation(X_training, y_training)

    accuracy = []
    testing_accuracy = []
    testing_f1 = []

    for var in params[indicator]:

        model = MLP(layer_sizes)

        bs = batch_size if indicator != "batch_size" else var
        lr = learning_rate if indicator != "learning_rate" else var
        ne = num_epoches if indicator != "num_epoches" else var
        
        accuracy.append(validator.cross_validation(5, model, bs, lr, ne))
        mlp_model = MLP(layer_sizes)
        mlp_model.training(scaled_X_training, y_training, scaled_X_testing, y_testing, batch_size=bs, learning_rate=lr, num_epoches=ne)
        acc, testing_f1_scores, f1_score = mlp_model.testing(scaled_X_testing, y_testing, is_f1=True)
        testing_accuracy.append(acc)
        testing_f1.append(f1_score)
        

    plt.plot(params[indicator], accuracy, color='blue', label='cross validation')
    plt.plot(params[indicator], testing_f1, color='green', label='cross validation')
    plt.plot(params[indicator], testing_accuracy, color='red', label='testing accuracy')

    plt.legend()
    plt.show()

def fit_MLP_and_plot_testing_vs_training(layer_sizes, scaled_X_training, y_training, scaled_X_testing, y_testing, batch_size=5, learning_rate=0.001, num_epoches=200):

    mlp_model = MLP(layer_sizes)

    mlp_model.training(scaled_X_training, y_training, scaled_X_testing, y_testing, batch_size=batch_size, learning_rate=learning_rate, num_epoches=num_epoches)

    plt.plot(range(1, len(mlp_model.training_accuracy_per_epoch) + 1), mlp_model.training_accuracy_per_epoch, color='blue', label='training accuracy')
    plt.plot(range(1, len(mlp_model.testing_accuracy_per_epoch) + 1), mlp_model.testing_accuracy_per_epoch, color='red', label='testing accuracy')

    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
