import numpy as np
from DataStorage.TimeSeriesDataset import VAEDataset

class FlattenedSequence:

    def __init__(self, training_data, validation_data, testing_data):

        self.flattened_training_data, self.torch_flattened_training_data = self.get_flattened_sequences(training_data)
        self.flattened_validation_data, self.torch_flattened_validation_data = self.get_flattened_sequences(validation_data)
        self.flattened_testing_data, self.torch_flattened_testing_data = self.get_flattened_sequences(testing_data)


    def get_flattened_sequences(self, data):

        X =  np.stack(data["time_serie"])
        X_flat = X.reshape(X.shape[0], -1) 
        y =  np.stack(data["response"])
        y_flat = y.reshape(y.shape[0], -1) 

        # concatenate the imput and ouput to a single vector
        result = np.concatenate([X_flat, y_flat], axis=1)

        #check the shape of each sequence
        print(result.shape)

        return result, VAEDataset(result)
