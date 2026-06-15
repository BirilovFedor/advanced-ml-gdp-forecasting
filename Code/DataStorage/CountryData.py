from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import math


class CountryData:

    def __init__(self, country, raw_dataframe):
        # name of a country
        self.country = country
        # the raw dataset
        self.raw_dataframe = raw_dataframe
        # names of current columns in the dataset
        self.features = self.raw_dataframe.columns.drop(['date'])
        # array containing sequence of 5 years period:
        self.sequence_dataframe = self.get_sequence_dataframe()
        # aggregated_sequence_dataframe
        self.aggregated_sequence_dataframe = self.get_aggregated_sequence_dataframe()
        # embedding vector which contain all feature except gdp, as average values of 5 year windows
        self.aggregated_embedding = self.get_embedding()
        # log transformed average GDP of a country
        self.log_average_GDP = math.log(self.aggregated_sequence_dataframe['GDPpc_2017$'].mean())
        # the marker for the country “Under-developed”, “Developing”, “Emerging”, or “Developed”
        self.development_marker = 0

    def get_sequence_dataframe(self):

        '''
        Docstring for get_sequence_dataframe

        create sequences of 5-year windows
        that overlap by 4 years (i.e., shift the window by 1 year at a time).
        
        :param self: Description

        :return array, where each element present a 5 year window with all features
        '''

        # to get array sequences of 5-year windows that overlap by 4 years (i.e., shift the window by 1 year at a time)    
        sequence = []
        for iter in range(0, len(self.raw_dataframe["date"]) - 4):
            sequence.append(self.raw_dataframe.iloc[iter: iter+5])
        return sequence
    
    def get_aggregated_sequence_dataframe(self):

        '''
        Docstring for get_aggregated_sequence_dataframe
        
        :param self: Description

        function is aggregating exitsing sequences of 5 years, taking average values for year year


        :return aggregated_vector: it is a single vector of numbers, that contain 5 sequnces of all features,
        where 1st is average of each feature for first year of each sequence
        where 2nd is average of each feature for second year of each sequence
        [1st_avg_feature1, 1st_avg_feature2, ..., 1st_avg_feature_n, 2nd_avg_feature1, ..., 2nd_avg_feature2, ..., 5th_avg_feature_n]

        :return pd.DataFrame(aggregated_vector.reshape(5, -1), columns=feature_columns): returns aggregated_vector in pandas dataframe in a following format
        [feature1_year1, feature2_year1, ..., feature_n_year1]
        [feature1_year2, feature2_year2, ..., feature_n_year2]
        .
        .
        .
        [feature1_year3, feature2_year3, ..., feature_n_year3]

        '''

        feature_columns = self.features
        
        all_sequences = []
        for window in self.sequence_dataframe:
            all_sequences.append(window[feature_columns])

        all_sequences = np.array(all_sequences)

        num_sequences = all_sequences.shape[0]
        flattened_sequences = all_sequences.reshape(num_sequences, -1)
        aggregated_vector = np.mean(flattened_sequences, axis=0)
        return pd.DataFrame(aggregated_vector.reshape(5, -1), columns=feature_columns)


    def get_embedding(self):
        
        '''
        Docstring for get_embedding

        function is used to get vector representation of an aggregated pandas dataframe
        dropping the feature GDP
        '''

        features = [f for f in self.features if f != "GDPpc_2017$"]


        sequence = np.array(self.aggregated_sequence_dataframe[features]) 
        
        return sequence.reshape(1, -1).flatten()
