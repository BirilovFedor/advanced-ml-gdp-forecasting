import pandas as pd
import numpy as np
from DataStorage.CountryData import CountryData


def read_csv(path, sep):
    # function to read the data from the csv file
    return pd.read_csv(path, sep=sep)

class Dataset:

    def __init__(self, imputed_training_dataset_path, imputed_testing_dataset_path):
        # raw training dataset
        self.training_dataset = read_csv(imputed_training_dataset_path, ',')
        # raw testing dataset
        self.testing_dataset = read_csv(imputed_testing_dataset_path, ',')
        # array with unique country names from training dataset
        self.training_country_names = self.get_country_names(self.training_dataset)
        # array with unique country names from training dataset
        self.testing_country_names = self.get_country_names(self.testing_dataset)
        # dictionary with data by countries from training dataset
        self.training_country_dict = self.get_countries_data(self.training_dataset, self.training_country_names)
        # dictionary with data by countries from testing dataset
        self.testing_country_dict = self.get_countries_data(self.testing_dataset, self.testing_country_names)

    def get_country_names(self, dataset):
        # get unique list of countries from the raw dataframe

        return dataset['country'].unique()
    
    def get_countries_data(self, dataset, country_names):
        # get a dictionary with dataframes splitted by country names

        country_dict = {}
        for country in country_names:

            country_dict[country] = CountryData(country, dataset[dataset['country'] == country].drop(columns=['country']))

        return country_dict
    
    def get_embedding(self):
        # function that is used to create a pandas dataframe with countries and corresponding log transformed

        '''
        Docstring for get_embedding
        function that is used to create a pandas dataframe with countries ad corresponding log transformed gdp value. It is using the
        gdp values of a country do divide the countries into 4 cohorts. Using quantile binning. Thus the cohorts contain equal amount of data, 
        this ensures class balance for better accuracy in future modeling 
        
        :param self: Description
        '''

        training_rows = [{
                'country': obj.country,
                'embedding': obj.aggregated_embedding,
                'gdp': obj.log_average_GDP
        }
            for obj in self.training_country_dict.values()
        ]

        testing_rows = [{
                'country': obj.country,
                'embedding': obj.aggregated_embedding,
                'gdp': obj.log_average_GDP
        }
            for obj in self.testing_country_dict.values()
        ]

        training_df = pd.DataFrame(training_rows)
        testing_df = pd.DataFrame(testing_rows)

        # create bins from the training dataset
        training_df['label'], bins = pd.qcut(training_df['gdp'], q=4, labels=[ 0, 1, 2, 3], retbins=True)

        # update the lower and upper bounds of bins to ensure that all testing data will fit
        bins[-1] = float('inf')
        bins[0] = float('-inf')

        testing_df['label'] = pd.cut(testing_df['gdp'], bins=bins, labels=[0, 1, 2, 3])

        # store training and testing embeddings
        self.training_embedding = training_df.drop(['gdp'], axis=1)
        self.testing_embedding = testing_df.drop(['gdp'], axis=1)

    def __getitem__(self, country_name):
        # use [] to access the country_dict objects using the country name

        return self.country_dict[country_name]
