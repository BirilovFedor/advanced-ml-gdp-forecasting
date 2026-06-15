
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

def read_csv(path, sep):

        # function to read the data from the csv file
        dataset =  pd.read_csv(path, sep=sep)
        return dataset.sort_values(['country', 'date']).reset_index(drop=True)

def merge_train_test(train_path, test_path, sep):
    '''
    Docstring for merge_train_test

    :param train_path: path of the training data
    :param testing_data: path of the testing data
    :param sep: separator in the csv file

    function to read traing test data and combine into one dataset
    '''

    train = read_csv(train_path, sep)
    test = read_csv(test_path, sep)

    merged = pd.concat([train, test], ignore_index=True)
    merged = merged.drop_duplicates()
    merged = merged.sort_values(["country", "date"]).reset_index(drop=True)

    return merged

class Plots:

    def __init__(self, training_dataset_path, testing_dataset_path, sep):

        '''
        Docstring for __init__

        :param train_path: path of the training data
        :param testing_data: path of the testing data
        :param sep: separator in the csv file

        merge training and testing data into a single dataframe to later make plots 
        '''
        self.dataset = merge_train_test(training_dataset_path, testing_dataset_path, sep)

    def plot_each_feature(self, features, output_folder = '', countries = ['United States', 'Brazil', 'Russian Federation', 'China']):
        
        '''
        Docstring for plot_each_feature

        :param features: features to plot
        :param output_folder: folder to store all plots
        :param sep: array containing all country names for countries that have to be plotted

        Function to plot all features for all selected countries and saved each file in the selected folder
        '''
        
        data_filtered = self.dataset[self.dataset["country"].isin(countries)]

        os.makedirs(output_folder, exist_ok=True)

        for country in data_filtered["country"].unique():
            country_data = data_filtered[data_filtered["country"] == country]

            for feature in features:
                if feature not in data_filtered.columns:
                    print(f"Skipping {feature}: not found in dataset")
                    continue
                
                country_data["date"] = pd.to_datetime(country_data["date"])

                plt.plot(country_data["date"], country_data[feature], marker="o")

                plt.figure(figsize=(10, 6),dpi=600)
                plt.plot(country_data["date"], country_data[feature], marker="o")
                plt.title(f"{feature} - {country}")
                plt.xlabel("Date")
                plt.ylabel(feature)                
                plt.tight_layout()
                ax = plt.gca()
                ax.xaxis.set_major_locator(mdates.YearLocator(5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.xticks(rotation=45)
                save_feature = feature.replace("/", "_").replace("\\", "_")
                file_name = f"{country}_{save_feature}.png"
                file_path = os.path.join(output_folder, file_name)

                plt.savefig(file_path, dpi=600, bbox_inches="tight")
                plt.close()

                print(f"Saved: {file_path} for {country} for {feature}")