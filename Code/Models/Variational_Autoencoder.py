import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler    
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(123)

def tensor_to_numpy_safe(tensor):
    """
    Safe conversion from PyTorch tensor to NumPy array.
    This is a trick created to avoids direct .numpy() that generates a bug in my environment
    Certainly due to incompatibility between my versions of PyToch and Numpy
    """
    return np.array(tensor.detach().cpu().tolist(), dtype=np.float32)

class Encoder(nn.Module):

    def __init__(self, encoder_specifications, input_dim, latent_dim):

        '''
        Docstring for __init__

        :param encoder_specifications: array of dicts, where each dict must contain 2 parameters,
        linear_ouput - size of the output from the linear block, is_relu - True if you want to 
        add non-linear Relu activation function after Linear block
        :param input_dim: dimention of the input data to the encoder
        :param latent_dim: dimention of the latent dataset 
        '''

        # initialize the class nn.Module
        super().__init__()

        Encoder_blocks = []

        linear_input = input_dim

        # ──────────────────────── Build Encoder block from Linear and Relu blocks ───────────────────────────────────────────────────
        for configuration in encoder_specifications:
            linear_ouput = configuration["linear_ouput"]
            is_relu = configuration["is_relu"]
            
            Encoder_blocks.append(nn.Linear(linear_input, linear_ouput))
            
            if is_relu:
                Encoder_blocks.append(nn.ReLU())

            linear_input = linear_ouput

        # store the output size from the the last Linear Block
        last_output = linear_input
        # ──────────────────────── Create whole Encoder sequence ───────────────────────────────────────────────────
        self.encoder = nn.Sequential(*Encoder_blocks)

        #──────────────────────── initialize mean of a latent distribution───────────────────────────────────────────────────
        self.mu = nn.Linear(last_output, latent_dim)
        #──────────────────────── initialize log variance of a latent distribution ───────────────────────────────────────────────────
        self.logvar = nn.Linear(last_output, latent_dim)
        
    def forward(self, X):
        '''
        Docstring for forward

        :param X: data that has to be encoded size [batch, input_dim]
        '''
            
        # apply the encoder sequential block to the input data
        X = self.encoder(X)

        # get mean of the latent distribution
        mu = self.mu(X)
        # get variance of the latent distribution
        log_var = self.logvar(X)

        return mu, log_var

class Decoder(nn.Module):

    def __init__(self, decoder_specifications, input_dim, latent_dim):

        '''
        Docstring for __init__

        :param decoder_specifications: array of dicts, where each dict must contain 2 parameters,
        linear_ouput - size of the output from the linear block, is_relu - True if you want to 
        add non-linear Relu activation function after Linear block. The last Linear block to get
        the imput dimention must not be added here, it is implemented manually
        :param input_dim: dimention of the input data to the decoder
        :param latent_dim: dimention of the latent dataset 
        '''

        # initialize the class nn.Module
        super().__init__()

        Decoder_blocks = []

        linear_input = latent_dim

        # ──────────────────────── Build Encoder block from Linear and Relu blocks ───────────────────────────────────────────────────
        for configuration in decoder_specifications:
            linear_ouput = configuration["linear_ouput"]
            is_relu = configuration["is_relu"]
            
            Decoder_blocks.append(nn.Linear(linear_input, linear_ouput))
            
            if is_relu:
                Decoder_blocks.append(nn.ReLU())

            linear_input = linear_ouput


        # store the output size from the the last Linear Block
        last_output = linear_input
        #──────────────────────── add linear block to get input dimetion───────────────────────────────────────────────────
        Decoder_blocks.append(nn.Linear(last_output, input_dim))

        # ──────────────────────── Create whole Encoder sequence ───────────────────────────────────────────────────
        self.decoder = nn.Sequential(*Decoder_blocks)
        
    def forward(self, Z):
        '''
        Docstring for forward

        :param Z: data that has to be decode size [batch, latent_dim]
        '''
          
        return self.decoder(Z)

class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim, encoder_specifications, decoder_specifications):
        
        # initialize the class nn.Module
        super().__init__()
        self.latent_dim = latent_dim
        #──────────────────────── initialise encoder block that return mu and log variance of the latent distribution ───────────────────────────────────────────────────
        self.encoder_model = Encoder(encoder_specifications, input_dim, latent_dim) 
         #──────────────────────── initialise decoder block that sample new data from the latend space ───────────────────────────────────────────────────
        self.decoder_model = Decoder(decoder_specifications, input_dim, latent_dim)

    def forward(self, X):
        '''
        Docstring for forward

        :param X: initial data input

        Function is encoding the data to get mean and variance of the latent distribution
        sample from the latent distribution, decode the values from the latent data to get 
        new observations
        '''

        mu, logvar = self.encoder_model(X)
        Z = self.sample_latent(mu, logvar)
        X_sample=self.decoder_model(Z)
        return X_sample, mu, logvar

    def sample_latent(self, mu, logvar):

        '''
        Docstring for sample_latent

        :param mu: mean of the distribution
        :param logvar: log variance of the distribution

        Sample from the distribtion z~N(mu, var)
        '''
        # get the standart deviation of a distribution
        std = torch.exp(0.5*logvar)
        # get random noise with same scale as standart deviation
        epsilon = torch.rand_like(std)
        # sample z from a distribution
        z = mu + epsilon *std

        return z 
    
    def vae_loss_function(self, x_hat, x, mu, logvar, beta=0.2):
        rec_loss = self.reconstruction_loss(x_hat, x)
        kl_div = self.kl_divergence(mu, logvar)
        total_loss = rec_loss + kl_div * beta
        return total_loss, rec_loss, kl_div

    def reconstruction_loss(self, x_hat, x):
        return nn.functional.mse_loss(x_hat, x, reduction='mean')
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def fit(self, train_data, num_epoches, batch_size, optimizer):

        '''
        Docstring for fit

        :param train_data: parameter of type torch tensor that contain both X and y for the training dataset
        :param num_epoches: parameter that controls the number of epoches during the training process
        :param batch_size: parameter to control size of a batch that feed to the model at a single step
        :param optimizer: used to update weights of a modelto minimize the used loss function, with a preset learning rate
        :param criterion: loss function that is used as model evaluation metrics in learning process
        '''
    
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
       
        total_losses = []
        recon_losses = []
        kl_losses = []
        # set the model to the training mode
        self.train()

        for epoch in range(num_epoches):
            running_total = 0.0
            running_recon = 0.0
            running_kl = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # ──────────────────────── Forward Block ────────────────────────
                #run the forward pass
                output, mu, logvar = self.forward(batch)

                # return infinity as a loss metric if the model exploaded
                if (
                torch.isnan(output).any() or torch.isinf(output).any() or
                torch.isnan(mu).any() or torch.isinf(mu).any() or
                torch.isnan(logvar).any() or torch.isinf(logvar).any()
                ):
                    print(f"Training stopped: NaN/Inf detected in forward pass at epoch {epoch+1}, batch {batch_idx+1}")
                    return [float('inf')], [float('inf')], [float('inf')]

                #get the loss
                loss, recon_loss, kl_div = self.vae_loss_function(output, batch, mu, logvar)
                # return infinity as a loss metric if the model exploaded
                if (
                torch.isnan(loss).any() or torch.isinf(output).any() or
                torch.isnan(recon_loss).any() or torch.isinf(mu).any() or
                torch.isnan(kl_div).any() or torch.isinf(logvar).any()
                ):
                    print(f"Training stopped: NaN/Inf detected in forward pass at epoch {epoch+1}, batch {batch_idx+1}")
                    return [float('inf')], [float('inf')], [float('inf')]
                
                # ──────────────────────── Backpropagation Block ────────────────────────
                #reset the gradients after reach batch
                optimizer.zero_grad()
                # compute gradients accroding to the loss
                loss.backward()

                for param in self.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Training stopped: NaN/Inf detected in gradients at epoch {epoch+1}, batch {batch_idx+1}")
                            return [float('inf')], [float('inf')], [float('inf')]


                # update weight accroding to the computed gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_local = batch.size(0)
                running_total += loss.item() * batch_size_local
                running_recon += recon_loss.item() * batch_size_local
                running_kl += kl_div.item() * batch_size_local

            epoch_total = running_total / len(train_loader.dataset)
            epoch_recon = running_recon / len(train_loader.dataset)
            epoch_kl = running_kl / len(train_loader.dataset)

            total_losses.append(epoch_total)
            recon_losses.append(epoch_recon)
            kl_losses.append(epoch_kl)

            #print(
                #f"[VAE] Epoch {epoch+1}/{num_epoches}, "
                #f"Total: {epoch_total:.4f}, Recon: {epoch_recon:.4f}, KL: {epoch_kl:.4f}"
            #)
        return total_losses, recon_losses, kl_losses
    
    def get_vae_latent_means(self, X):

        '''
        Docstring for get_vae_latent_means

        :param X: the flattened sequence in either float or numpy, NOT TENSOR INPUT

        function is generating the mu and log variance for each input based on the pretrained model
        '''

        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            mu, logvar = self.encoder_model(X_tensor)
        return tensor_to_numpy_safe(mu), tensor_to_numpy_safe(logvar)
    
    def reduce_mean_dim(self,latent_mu,  n_components, perplexity=40, max_iter=1000,  random_state=42, verbose=0):
        
        '''
        Docstring for reduce_mean 
        '''

        tsne   = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter,
              random_state=random_state, verbose=verbose)
        return tsne.fit_transform(latent_mu)

    def decode(self, X):

        return self.decoder_model(X)

    def generate_syntetic_data(self, num_samples):

        '''
        Docstring for generate_syntetic_data

        :param num_samples: number of samples to generate
        
        this function is used to generate some data from the latend distribution of a pretrained model
        '''
        
        self.eval() 

        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            synthetic_scaled = self.decode(z)
        return tensor_to_numpy_safe(synthetic_scaled)