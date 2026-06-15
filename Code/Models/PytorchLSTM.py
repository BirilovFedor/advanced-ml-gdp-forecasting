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

class Pytorch_LSTM(nn.Module):

    '''
    Docstring for Pytorch_LSTM realization

    :param nn.Module: we define class that inherits from the nn.Module, this gives all Pytorch functions to our class
    '''
    
    def __init__(self, hidden_size, num_layers, num_features, output_size, drop_out):

        '''
        Dosctring for __init__


        :param hidden_size: parameter used to identify the hidden size of the LSTM model
        :param num_layers: parameter used to identify the number of stackes LSTM models
        :param num_features: parameter used to identify the initial number of parameters
        :param num_features: parameter used to identify the output_size (how many years want to predict)
        :param drop_out: parameter that controls the ratio of turned-off neurons
        '''


        # run the constructor for the nn.Module from pytorch
        super(Pytorch_LSTM, self).__init__()
        self.num_features = num_features
        # define device to use gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ──────────────────────── Build LSTM block ───────────────────────────────────────────────────     

        self.LSTM_model = nn.LSTM(input_size=num_features,
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, batch_first=True,
                                        dropout=drop_out if num_layers > 1 else 0.0)

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
        out, _ = self.LSTM_model(X)
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