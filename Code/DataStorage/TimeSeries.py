
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler    
from DataStorage.TimeSeriesDataset import TimeSeriesDataset

def read_csv(path, sep, response_variable, is_log=False):
    '''
    Docstring for read_csv
    function to read the data from the csv file

    :param path: path of the file to read the data from
    :param sep: separator between instances in csv file
    :param response_variable: prediction target variable
    :param is_log: False by default, set to True if want to apply log transformation to the target variable
    '''
    
    df = pd.read_csv(path, sep=sep)
    if is_log:
        df.loc[:, response_variable]=np.log(df[response_variable])
    return df

class TimeSeriesCountry:

    '''
    Docstring for __init__

    :param training_data: training data
    :param validation_data: validation data
    :param testing_data: testing data
    :param country_name: name of a country
    :param features: features that have to be used in modeling process
    :param response_variable: prediction target variable
    :param window_length: number of historical years to use in predictions
    :param test_size: number of years to predict
    '''

    def __init__(self, training_data, validation_data, testing_data, country_name, features, response_variable, window_length, test_size):
        self.training_data = training_data
        self.validation_data = validation_data
        self.testing_data = testing_data
        self.country_name = country_name
        self.window_length = window_length
        self.test_size = test_size
        self.features = features
        self.response_variable = response_variable
        # Standart Scaler for this country
        self.scaler = self.standartize_data()
        # Get Windows for this country, separated by train, val, test
        self.training_time_series, self.validation_time_series, self.testing_time_series = self.get_windows()

    def standartize_data(self):
        '''
        Docstring for standartize_data

        Apply standart scaling to training, validation and testing data, for each country separately
        '''

        scaler = StandardScaler()

        self.training_data.loc[:, self.features] = scaler.fit_transform(self.training_data[self.features])

        self.validation_data.loc[:, self.features] = scaler.transform(self.validation_data[self.features])

        self.testing_data.loc[:, self.features] = scaler.transform(self.testing_data[self.features])

        return scaler

    def get_windows(self):

        '''
        Docstring for get_windows

        Function is creating a pandas dataframe with country name, start date, end date,  time_series = input window, reponse = output window for training, validation and testing data
        creating sequences of length input_window and output window for modeling
        '''

        time_series = []
        for iter in range(len(self.training_data) - (self.window_length + self.test_size) + 1):
            time_series.append({
                "country": self.country_name,
                "start_date": self.training_data["date"].iloc[iter],
                "end_date": self.training_data["date"].iloc[iter+self.window_length-1],
                "time_serie": self.training_data[self.features].iloc[iter:(iter+self.window_length)].to_numpy(),
                "response": self.training_data[self.response_variable].iloc[(iter+self.window_length):(iter+self.window_length+self.test_size)].to_numpy()
            }) 
        training_time_series = pd.DataFrame(time_series)
        time_series = []
        dummy = pd.concat([self.training_data, self.validation_data], ignore_index=True)
        for iter in range(len(self.validation_data) - self.test_size,-1, -1):
            time_series.append({
                "country": self.country_name,
                "start_date": dummy["date"].iloc[-(self.window_length+self.test_size+iter)],
                "end_date": dummy["date"].iloc[-(1+self.test_size+iter)],
                "time_serie": dummy[self.features].iloc[-(self.window_length+self.test_size+iter):-(self.test_size+iter)].to_numpy(),
                "response": dummy[self.response_variable].iloc[-(self.test_size+iter): None if iter ==0 else -(iter)].to_numpy()
            }) 
        validation_time_series = pd.DataFrame(time_series)

    
        time_series = []
        dummy = pd.concat([self.training_data, self.validation_data, self.testing_data], ignore_index=True)
        for iter in range(len(self.testing_data) - self.test_size, -1, -1):
            time_series.append({
                "country": self.country_name,
                "start_date":  dummy["date"].iloc[-(self.window_length+self.test_size+iter)],
                "end_date": dummy["date"].iloc[-(1+self.test_size+iter)],
                "time_serie": dummy[self.features].iloc[-(self.window_length+self.test_size+iter):-(self.test_size+iter)].to_numpy(),
                "response": dummy[self.response_variable].iloc[-(self.test_size+iter): None if iter ==0 else -(iter)].to_numpy()
            })
        testing_time_series = pd.DataFrame(time_series)
        
        return training_time_series, validation_time_series, testing_time_series


class TimeSeries:

    def __init__(self, imputed_training_dataset_path, imputed_validation_dataset_path, imputed_testing_dataset_path, features, response_variable, test_size, window_length):

        '''
        Docstring for __init__

        :param imputed_training_dataset_path: path to the training dataset
        :param imputed_validation_dataset_path: path to the validation dataset
        :param imputed_testing_dataset_path: path to the testing dataset
        :param features: features that supposed to be used in the time series forecast
        :param response_variable: target for predictions
        :param test_size: number of years to predict
        :param window_length: number of historical years used in prediction
        '''

        # training data
        self.training_data = read_csv(imputed_training_dataset_path, ',', response_variable, True)
        # validation data
        self.validation_data = read_csv(imputed_validation_dataset_path, ',', response_variable, True)
        # testing data
        self.testing_data = read_csv(imputed_testing_dataset_path, ',', response_variable, True)
        # unique country names
        self.unique_countries = self.training_data['country'].unique()
        # features of the dataframe
        self.features = features
        self.window_length = window_length
        self.test_size = test_size
        # response varibale
        self.response_variable = response_variable
        # dictionary where key is the country name and the value is the TimeSeriesCountry object
        self.countries_dict = self.get_countries_data(window_length, test_size)
        # time series dataframe and torch object for training data
        self.training_time_series, self.torch_training_time_series = self.get_training_windows()
        # time series dataframe and torch object for validation data
        self.validation_time_series, self.torch_validation_time_series = self.get_validation_window()
        # time series dataframe and torch object for testing data
        self.testing_time_series, self.torch_testing_time_series = self.get_testing_window()
    
    def get_countries_data(self, window_length, test_size):

        '''
        Docstring for get_countries_data

        Function to iterate over all countries and create variable self.countries_dict
        '''
        
        country_dict = {}
        for country in self.unique_countries:
            
            country_data = TimeSeriesCountry(self.training_data[self.training_data["country"] == country],
                                             self.validation_data[self.validation_data["country"] == country],
                                             self.testing_data[self.testing_data["country"] == country], country, 
                                             self.features, self.response_variable, window_length, test_size)
            country_dict[country] = country_data
        return country_dict

    def get_training_windows(self):
        '''
        function to stack training data from all countries into a single dataframe

        :return time_series: pandas dataframe with all training data from all countries
        :return  TimeSeriesDataset(np.stack(time_series["time_serie"]), np.stack(time_series["response"])): is the torch class for the time_series
        '''

        dfs = [obj.training_time_series for obj in self.countries_dict.values()]
        
        time_series = pd.concat(dfs, ignore_index=True)

        X = np.stack(time_series["time_serie"])
        y = np.stack(time_series["response"])

        print(54*"#")
        print(f"SHAPE FOR THE TRAINING X DATA: {X.shape}")
        print(f"SHAPE FOR THE TRAINING y DATA: {y.shape}")
        print(54*"#")
            
        return time_series, TimeSeriesDataset(X, y)

    def get_validation_window(self):
        '''
        function to stack validation data from all countries into a single dataframe

        :return time_series: pandas dataframe with all validation data from all countries
        :return  TimeSeriesDataset(np.stack(time_series["time_serie"]), np.stack(time_series["response"])): is the torch class for the time_series
        '''

        dfs = [obj.validation_time_series for obj in self.countries_dict.values()]
        
        time_series = pd.concat(dfs, ignore_index=True)
        X = np.stack(time_series["time_serie"])
        y = np.stack(time_series["response"])

        print(54*"#")
        print(f"SHAPE FOR THE VALIDATION X DATA: {X.shape}")
        print(f"SHAPE FOR THE VALIDATION y DATA: {y.shape}")
        print(54*"#")
            
        return time_series, TimeSeriesDataset(X, y)
    
    def get_testing_window(self):
        '''
        function to stack testing data from all countries into a single dataframe

        :return time_series: pandas dataframe with all testing data from all countries
        :return  TimeSeriesDataset(np.stack(time_series["time_serie"]), np.stack(time_series["response"])): is the torch class for the time_series
        '''

        dfs = [obj.testing_time_series for obj in self.countries_dict.values()]
        
        time_series = pd.concat(dfs, ignore_index=True)

        X = np.stack(time_series["time_serie"])
        y = np.stack(time_series["response"])

        print(54*"#")
        print(f"SHAPE FOR THE TESTING X DATA: {X.shape}")
        print(f"SHAPE FOR THE TESTING y DATA: {y.shape}")
        print(54*"#")
            
        return time_series, TimeSeriesDataset(X, y)
    
    def get_single_testing(self, country_name):
        '''
        fuction to get single testing point using country_name 

        :return TimeSeriesDataset(np.stack([time_series.testing_time_series["time_serie"].iloc[-1]]), np.stack([time_series.testing_time_series["response"].iloc[-1]])):  is the torch class for the testing point
        :return  time_series.scaler: is the scaler used for this country

        '''

        time_series = self.countries_dict[country_name]
        return TimeSeriesDataset(np.stack([time_series.testing_time_series["time_serie"].iloc[-1]]), np.stack([time_series.testing_time_series["response"].iloc[-1]])), time_series.scaler
    

    def add_simulated_data(self, samples):

        X = samples[:, :-self.test_size]
        y = samples[:, -self.test_size:]

        X = X.reshape(X.shape[0], self.window_length, len(self.features))
        y = y.reshape(y.shape[0], self.test_size)

        old_X = np.stack(self.training_time_series["time_serie"].copy())
        old_y = np.stack(self.training_time_series["response"].copy())

        new_X = np.concatenate([old_X, X], axis=0)
        
        new_y = np.concatenate([old_y, y], axis=0)

        self.torch_sim_training_time_series = TimeSeriesDataset(new_X, new_y)
        

