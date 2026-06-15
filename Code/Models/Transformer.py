import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler    
import numpy as np
import math
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


class TimeSeriesEmbedding(nn.Module):

    def __init__(self, num_features, embedding_length):

        '''
        Docstring for __init__

        :param num_features: initial number of features in the input sequence
        :param embedding_length: number of feature after the embedding

        this function is used to increase number of features of a model
        '''

        super(TimeSeriesEmbedding, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Linear(num_features, embedding_length)

    def forward(self, X):

        X = self.embedding(X)
        return X

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 500):

        '''
        Docstring for __init__

        :param max_len: max length of positional encoding that can be handled by the encoder
        :param d_model: number of feature after the embedding

        this function is used to create max_length x d_model matrix of positional encoding

        '''

        super(SinusoidalPositionalEncoding, self).__init__()
        
        # create matrix size of max_length x d_model fill it with zeroes
        pos_encode =  torch.zeros(max_len, d_model)
        # create positional matrix size max_length x 1 time time steps
        position = torch.arange(0, max_len).unsqueeze(1).float()
        #division term in Sinusoidal postional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply formula for all even positions 
        pos_encode[:, 0::2] = torch.sin(position*div_term)
        # apply formula for all odd positions 
        pos_encode[:, 1::2]= torch.cos(position*div_term)

        # reshape to  (1, max_len, d_model) to be able to add directly 
        pos_encode = pos_encode.unsqueeze(0)

        # store positional encoding in a buffer so it is updated during training
        self.register_buffer('pos_encode', pos_encode)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        return x + self.pos_encode[:, :x.size(1), :]


class Transformer(nn.Module):

    def __init__(self, num_features, d_model, nhead, dim_feedforward, dropout, num_layers, output_size):

        '''
        Docstring for __init__

        :param num_features: initial number of features
        :param d_model: number of feature after the embedding
        :param nhead: number of attention head in multi-head attention
        :param dim_feedforward: hidden layer sizeof feedforward network
        :param dropout: fraction of neurons to turn-off
        :param num_layers: number of stacked encoder layers
        :param output_size: output size
        '''

        super(Transformer, self).__init__()
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ──────────────────────── Create Linear Trasnformation to get the embedding of initial sequence───────────────────────────────────────────────────
        self.embedding = TimeSeriesEmbedding(num_features, d_model)
        # ──────────────────────── Create Positional Encoder ───────────────────────────────────────────────────
        self.positional_encoder = SinusoidalPositionalEncoding(d_model, max_len=500)

        # ──────────────────────── Create Transformer Encoder Layer Instance───────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, 
                                                    batch_first=True
                                                )
        # ──────────────────────── Create Transformer Encoder Object with num_layers───────────────────────────────────────────────────
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, output_size)
        self.to(self.device)

    def forward(self, X):
        # size of X(batch, input_length, num_features)
        
        # embedding (batch, input_length, num_features)
        X = self.embedding(X)
        # positional encoding (batch, input_length, num_features)
        X = self.positional_encoder(X)
        # transformer encoder (batch, input_length, num_features)
        X = self.transformer_encoder(X)
        # calculate mean poling across time (batch, num_features)
        X = X[:, -1, :]#X.mean(dim=1)

        out = self.output(X)
        return out
    

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