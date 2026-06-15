import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler    
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(123)

class CNN(nn.Module):

    def __init__(self, cnn_layers, num_features, drop_out):

        '''
        Dosctring for __init__

        :param cnn_layers: dict that store information for each of the CNN layers:
            1. filters_num - number of filters
            2. kernel_size - filter size
            3. padding_size - number of zeroes around original data
            4. stride_size - step size for a filter
            5. pool_type - can take avg, max and adaptive pool types
            6. pool_size - number of elements across time that are considered in the same time

        :param num_features: parameter used to identify the initial number of parameters
        :param drop_out: parameter that controls the ratio of turned-off neurons
        '''

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CNN_blocks = []

        in_channels = num_features

        # ──────────────────────── Build CNN block ───────────────────────────────────────────────────

        for configuration in cnn_layers:
            out_channels = configuration["filters_num"]
            kernel_size = configuration["kernel_size"]
            padding_size = configuration["padding_size"]
            stride_size = configuration["stride_size"]
            pool_type = configuration["pool_type"]
            pool_size = configuration["pool_size"]
            
            # append the Convolutional Block
            CNN_blocks.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size, stride=stride_size))

            # append the RELU block, max function
            CNN_blocks.append(nn.ReLU())

            # append the pooling block if it is necessary
            if pool_type == "adaptive":
                CNN_blocks.append(nn.AdaptiveAvgPool1d(output_size = pool_size))
            if pool_type == "max":
                CNN_blocks.append(nn.MaxPool1d(kernel_size = pool_size))
            if pool_type == "avg":
                CNN_blocks.append(nn.AvgPool1d(kernel_size = pool_size))

            in_channels = out_channels

        self.final_out = in_channels

        #append drop out block if it it used to randomly turn of some of the neurons to prevent overfitting
        if drop_out > 0:
            CNN_blocks.append(nn.Dropout(drop_out))
        self.CNN_model = nn.Sequential(*CNN_blocks)

    def forward(self, X):
        '''
        Docstring for forward_pass:

        runs a forward pass for the CNN model

        :param X: initial data with size (batch, length_input_window, num_features)

        '''
        X = X.to(self.device)
        # run the CNN model to extract features
        X = self.CNN_model(X)
        return X


class LSTM(nn.Module):
     

    def __init__(self, hidden_size, num_layers, input_size, drop_out):

        '''
        Dosctring for __init__


        :param hidden_size: parameter used to identify the hidden size of the LSTM model
        :param num_layers: parameter used to identify the number of stackes LSTM models
        :param num_features: parameter used to identify the initial number of parameters
        :param output_size: parameter used to identify the output_size (how many years want to predict)
        :param drop_out: parameter that controls the ratio of turned-off neurons
        '''


        # run the constructor for the nn.Module from pytorch
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ──────────────────────── Build LSTM block ───────────────────────────────────────────────────     

        self.LSTM_model = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, batch_first=True,
                                        dropout=drop_out if num_layers > 1 else 0.0)

    def forward(self, X):
        '''
        Docstring for forward_pass:

        runs a forward pass for the LSTM model

        :param X: output from the CNN with size (batch, input_size, num_features)

        '''
        out, _ = self.LSTM_model(X)
        return out


class CNN_LSTM(nn.Module):

    '''
    Docstring for CNN LSTM realization

    :param nn.Module: we define class CNN that inherits from the nn.Module, this gives all Pytorch functions to our class
    '''
    
    def __init__(self, cnn_layers, hidden_size, num_layers, num_features, output_size, drop_out):

        '''
        Dosctring for __init__

        :param cnn_layers: dict that store information for each of the CNN layers:
            1. filters_num - number of filters
            2. kernel_size - filter size
            3. padding_size - number of zeroes around original data
            4. stride_size - step size for a filter
            5. pool_type - can take avg, max and adaptive pool types
            6. pool_size - number of elements across time that are considered in the same time

        :param hidden_size: parameter used to identify the hidden size of the LSTM model
        :param num_layers: parameter used to identify the number of stackes LSTM models
        :param num_features: parameter used to identify the initial number of parameters
        :param output_size: parameter used to identify the output_size (how many years want to predict)
        :param drop_out: parameter that controls the ratio of turned-off neurons
        '''


        # run the constructor for the nn.Module from pytorch
        super(CNN_LSTM, self).__init__()
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ──────────────────────── Build CNN block using cnn_layers dict───────────────────────────────────────────────────

        self.CNN_model = CNN(cnn_layers=cnn_layers, num_features=num_features, drop_out=drop_out)


        # ──────────────────────── Build LSTM block ───────────────────────────────────────────────────     

        self.LSTM_model = LSTM(hidden_size=hidden_size,
                               num_layers=num_layers, input_size=self.CNN_model.final_out, drop_out=drop_out)

        # ──────────────────────── Build Output Block ───────────────────────────────────────────────────   
        self.output = nn.Linear(hidden_size, output_size)

        self.to(self.device)

    def forward(self, X):
        '''
        Docstring for forward_pass:

        runs a forward pass for the CNN LSTM model

        :param X: initial data with size (batch, length_input_window, num_features)

        '''
        X = X.to(self.device)
        # The CNN model expect the input to be (batch, num_features, length_input_window)
        X = X.permute(0, 2, 1)
        # run the CNN model to extract features
        X = self.CNN_model(X)
        # The LSTM model expect the input to be (batch, length_input_window, num_features)
        X = X.permute(0, 2, 1)
        out = self.LSTM_model(X)
        return self.output(out[:, -1, :])

    def fit(self, train_data, num_epoches, batch_size, optimizer, criterion, is_return=False):

        '''
        Docstring for fit

        :param train_data: parameter of type torch tensor that contain both X and y for the training dataset
        :param num_epoches: parameter that controls the number of epoches during the training process
        :param batch_size: parameter to control size of a batch that feed to the model at a single step
        :param optimizer: used to update weights of a modelto minimize the used loss function, with a preset learning rate
        :param criterion: loss function that is used as model evaluation metrics in learning process
        '''
        counter = 0
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        last_loss = 0
        for epoch in range(num_epoches):
            # set the model to the training mode
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                # ──────────────────────── Forward Block ────────────────────────
                #run the forward pass
                output = self.forward(batch_X)
                #get the loss
                loss = criterion(output, batch_y)
                
                # ──────────────────────── Backpropagation Block ────────────────────────
                #reset the gradients afte reach batch
                optimizer.zero_grad()
                # compute gradients accroding to the loss
                loss.backward()
                # update weight accroding to the computed gradients
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
            last_loss = train_loss / len(train_loader.dataset)
        if is_return:
            return last_loss

    def evaluate(self, data, batch_size, criterion):

        '''
        Docstring for evaluate
        :param data: validation data of a type torch tensor
        :param batch_size: size of batch fed to the model at a single iteration of a model
        :param criterion: loss function used to identify performance of a model

        return the loss for the whole validation dataset
        '''

        self.eval()

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        train_loss =0
        for batch_X, batch_y in test_loader:
            output = self.forward(batch_X)
            loss = criterion(output, batch_y)
            train_loss += loss.item() * batch_X.size(0)

        return train_loss / len(test_loader.dataset)

    def test_data(self, data, batch_size):

        '''
        Docstring for evaluate
        :param data: validation data of a type torch tensor
        :param batch_size: size of batch fed to the model at a single iteration of a model

        return the loss for the whole testing dataset in MSE and MAE metrics, not in the original scale
        '''

        self.eval()

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        criterion_MSE = nn.MSELoss()
        criterion_MAE = nn.L1Loss()
        train_loss_MSE =0
        train_loss_MAE =0
        for batch_X, batch_y in test_loader:
            output = self.forward(batch_X)
            loss_MSE = criterion_MSE(output, batch_y)
            loss_MAE = criterion_MAE(output, batch_y)
            train_loss_MSE += loss_MSE.item() * batch_X.size(0)
            train_loss_MAE += loss_MAE.item() * batch_X.size(0)

        return train_loss_MSE / len(test_loader.dataset), train_loss_MAE / len(test_loader.dataset)

    def predict(self, data, batch_size):

        '''
        Docstring for predict
        :param data: validation data of a type torch tensor
        :param batch_size: size of batch fed to the model at a single iteration of a model

        return the predictions and actual values
        '''

        self.eval()
        predictions, actuals = [], []

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for batch_X, batch_y in test_loader:
            outputs = self.forward(batch_X)
            predictions.append(outputs.squeeze().detach().numpy())
            actuals.append(batch_y.detach().numpy())
        predictions = np.concatenate(predictions).reshape(-1, 1)
        actuals = np.concatenate(actuals).reshape(-1, 1)
        return predictions, actuals

    def test_single_point(self, data, scaler):
        self.eval()
        predictions, actuals = [], []

        test_loader = DataLoader(data, batch_size=1, shuffle=False)
        for batch_X, batch_y in test_loader:
            outputs = self.forward(batch_X)
            predictions.append(outputs.squeeze().detach().numpy())
            actuals.append(batch_y.detach().numpy())
        predictions = np.concatenate(predictions).reshape(-1, 1)
        actuals = np.concatenate(actuals).reshape(-1, 1)
        return inverse_scale(predictions, scaler, self.num_features), inverse_scale(actuals, scaler, self.num_features)


def inverse_scale(values, scaler, num_features):
    '''
    Docstring for inverse_scale
    
    :param values: values that have to be inverse scaled
    :param scaler: scaler of this data to apply inverse scaling
    :param num_features: num of features in the whole dataset
    '''

    values_with_placeholders = np.hstack([values, np.zeros((values.shape[0], num_features-1))])
    # Apply inverse_transform
    values_inverse = scaler.inverse_transform(values_with_placeholders)

    inverse_scaled_values = values_inverse[:, 0]
    return np.exp(inverse_scaled_values)