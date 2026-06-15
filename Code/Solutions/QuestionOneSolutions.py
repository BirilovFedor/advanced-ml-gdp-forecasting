# ──────────────────────── import from the project ──────────────────────────────────────────────────
from Imputation.HybridImputer import HybridImputer
from EDA.Plots import Plots
from DataStorage.Dataset import Dataset

# ──────────────────────── import from external libraries ──────────────────────────────────────────────────
import datetime
import wbdata


class QuestionOneSolutions:


    def __init__(self, initial_dataset_path, imputed_training_dataset_path_country, 
                 imputed_testing_dataset_path_country, 
                 imputed_training_dataset_path_year,
                 imputed_validation_dataset_path_year,
                 imputed_testing_dataset_path_year,
                 plots_folder_path,
                 indicators, features, start_year = 1980, end_year = 2023):
        '''
        Docstring for __init__

        :param initial_dataset_path: the path there the initial dataset will be stored
        :param imputed_training_dataset_path_country: the path there the imputed training data for classification should be stored
        :param imputed_testing_dataset_path_country: the path there the imputer testing data for classification should be stored
        :param imputed_training_dataset_path_year: the path there the imputed training data for time series regression should be stored
        :param imputed_validation_dataset_path_year: the path there the imputed validation data for time series regression should be stored
        :param imputed_testing_dataset_path_year: the path there the imputer testing data for time series regression should be stored
        :param indicators: indicators that have to be downloaded from the world bank API
        :param features: numerical features that are used in the analysis process
        :param start_year: starting year of the data considered in the analysis, 1980 by default
        :param end_year: last year of the data considered in the analysis, 2023 by default
        '''

        self.initial_dataset_path = initial_dataset_path
        self.imputed_training_dataset_path_country = imputed_training_dataset_path_country
        self.imputed_testing_dataset_path_country = imputed_testing_dataset_path_country
        self.imputed_training_dataset_path_year = imputed_training_dataset_path_year
        self.imputed_validation_dataset_path_year = imputed_validation_dataset_path_year
        self.imputed_testing_dataset_path_year = imputed_testing_dataset_path_year
        self.plots_folder_path = plots_folder_path
        self.indicators = indicators
        self.features = features
        self.start_year = start_year
        self.end_year = end_year

    def solve_question(self):

        '''
        Docstring for solve_question

        runner for question one
        please uncomment the function you want to be runned

        THIS FUNCTION RETURN THE DATASET CLASS THAT CONTAIN THE SEQUENCES FOR Q2
        '''

        # ──────────────────────── Download the data from the worldbank ───────────────────────────────────────────────────
        '''QuestionOneSolutions.download_world_bank_data(indicators=self.indicators,
                                                      start_year=self.start_year,
                                                      end_year=self.end_year,
                                                      output_file=self.initial_dataset_path)'''
    
        # ──────────────────────── Explore the missing values and impute the data ───────────────────────────────────────────────────
        '''QuestionOneSolutions.imputation(features=self.features,
                                                initial_dataset_path=self.initial_dataset_path,
                                                imputed_training_dataset_path_country=self.imputed_training_dataset_path_country,
                                                imputed_testing_dataset_path_country=self.imputed_testing_dataset_path_country,
                                                imputed_training_dataset_path_year=self.imputed_training_dataset_path_year, 
                                                imputed_validation_dataset_path_year=self.imputed_validation_dataset_path_year, 
                                                imputed_testing_dataset_path_year=self.imputed_testing_dataset_path_year,  
                                                test_ratio=0.2,
                                                year_split_sizes=[10, 5])'''


        # ──────────────────────── Plot each features for list of required countries ───────────────────────────────────────────────────
        '''countires_to_plot = ['United States', 'Brazil', 'Russian Federation', 'China']
        self.plot_countries(features=self.features, 
                            imputed_training_dataset_path_country=self.imputed_training_dataset_path_country,
                            imputed_testing_dataset_path_country=self.imputed_testing_dataset_path_country,
                            countires_to_plot=countires_to_plot, folder_path=self.plots_folder_path)'''
        

        # ────────────────────────  Create Sequences for the MLP classificaiton ───────────────────────────────────────────────────
        dataset = QuestionOneSolutions.create_sequencies(imputed_training_dataset_path_country=self.imputed_training_dataset_path_country,
                                                      imputed_testing_dataset_path_country=self.imputed_testing_dataset_path_country)
        
        

        return dataset

    @staticmethod
    def create_sequencies(imputed_training_dataset_path_country, 
                          imputed_testing_dataset_path_country):
        
        '''
        Docstring for create_sequencies

        :param imputed_training_dataset_path_country: path to the training data
        :param imputed_testing_dataset_path_country: path to the testing data

        :return Dataset instance that contain array with country dicts that contain the sequences
        '''
        return Dataset(imputed_training_dataset_path=imputed_training_dataset_path_country, 
                        imputed_testing_dataset_path=imputed_testing_dataset_path_country)
        
    @staticmethod
    def plot_countries(features, imputed_training_dataset_path_country,
                       imputed_testing_dataset_path_country,
                       countires_to_plot, folder_path):
        '''
        Docstring for plot_countries
        
        :param features: features that have to be ploted
        :param imputed_training_dataset_path_country: training data path
        :param imputed_testing_dataset_path_country: testing data path
        :param countires_to_plot: contires that have to be plotted
        :param folder_path: folder to store all plots

        function to plot all features for all countries across whole timeline
        '''

        plots = Plots(training_dataset_path=imputed_training_dataset_path_country,
                      testing_dataset_path=imputed_testing_dataset_path_country, sep=',')
        
        plots.plot_each_feature(features, output_folder=folder_path, countries=countires_to_plot)

    @staticmethod
    def imputation(features, initial_dataset_path, 
                   imputed_training_dataset_path_country, 
                   imputed_testing_dataset_path_country, 
                   imputed_training_dataset_path_year,
                   imputed_validation_dataset_path_year,
                   imputed_testing_dataset_path_year,
                   test_ratio = 0.2,
                   year_split_sizes = [10, 5]):
        '''
        Docstring for imputation

        :param features: features that have to be imputed, only numerical features
        :param initial_dataset_path: path to the data that have to be imputed
        :param imputed_training_dataset_path_country: path to the file there imputed training data for classification should be saved 
        :param imputed_testing_dataset_path_country: path to the file there imputed testing data for classification should be saved
        :param imputed_training_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param imputed_validation_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param imputed_testing_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param test_ratio: ratio of the data saved for testing equal to 0.2 by default
        :param year_split_sizes: array containing 2 values, number of years for the validation and number of years for testing equal to [10, 5] by default
        '''

        # create imputer
        imputer = HybridImputer(path=initial_dataset_path, features=features, split_criteria='country')

        # analyse the data, amount of missing data, type of the data
        imputer.data_checks(data = imputer.data)

        # analyse the amount of missing data for a certain country subset
        imputer.missing_values_for_certain_countires(data = imputer.data, countries_names=['United States', 'Brazil', 'China', 'Russian Federation'])

        QuestionOneSolutions.imputation_country(features=features,
                                                initial_dataset_path=initial_dataset_path,
                                                imputed_training_dataset_path_country=imputed_training_dataset_path_country,
                                                imputed_testing_dataset_path_country=imputed_testing_dataset_path_country, 
                                                test_ratio=test_ratio)

        QuestionOneSolutions.imputation_year(features=features,
                                                initial_dataset_path=initial_dataset_path,
                                                imputed_training_dataset_path_year=imputed_training_dataset_path_year,
                                                imputed_validation_dataset_path_year=imputed_validation_dataset_path_year,
                                                imputed_testing_dataset_path_year=imputed_testing_dataset_path_year, 
                                                year_split_sizes=year_split_sizes)
    @staticmethod
    def download_world_bank_data(indicators, start_year, end_year, output_file):
        """
        Downloads tabular and time-series data from the World Bank API.

        Parameters:
            indicators (dict): A dictionary of indicators {"indicator_code": "indicator_name"}.
            country_codes (list): List of country codes to include in the data.
            start_year (int): Start year for the data.
            end_year (int): End year for the data.
            output_file (str): Path to save the downloaded data as a CSV file.

        Returns:
            None
        """

        expression='^(?!(?:World|Upper middle income|Sub\-Saharan Africa excluding South Africa|Sub\-Saharan Africa|Latin America \& Caribbean|IDA countries in Sub\-Saharan Africa classified as fragile situations|IDA\scountries\sin\sSub\-Saharan\sAfrica\snot\sclassified\sas\sfragile\ssituations|Sub\-Saharan Africa excluding South Africa and Nigeria|South Asia \(IDA \& IBRD\)|Sub-Saharan Africa \(IDA \& IBRD countries\)|Middle East \& North Africa \(IDA \& IBRD countries\)|Latin America \& the Caribbean \(IDA \& IBRD countries\)|Europe \& Central Asia \(IDA \& IBRD countries\)|East Asia \& Pacific \(IDA \& IBRD countries\)|Sub\-Saharan Africa|Sub\-Saharan Africa \(excluding high income\)|Resource rich Sub\-Saharan Africa countries|IDA countries not classified as fragile situations\, excluding Sub-Saharan Africa|Non\-resource rich Sub\-Saharan Africa countries|Middle East \& North Africa \(excluding high income\)|Middle East \(developing only\)|Latin America \& Caribbean \(excluding high income\)|IBRD\, including blend|Heavily indebted poor countries \(HIPC\)|IDA countries classified as fragile situations\, excluding Sub\-Saharan Africa|Europe \& Central Asia \(excluding high income\)|East Asia \& Pacific \(excluding high income\)|Sub\-Saharan Africa \(IDA\-eligible countries\)|IDA countries in Sub\-Saharan Africa classified as fragile situations|South Asia \(IDA\-eligible countries\)|IDA countries in Sub\-Saharan Africa not classified as fragile situations|Middle East \& North Africa \(IDA\-eligible countries\)|Latin America \& the Caribbean \(IDA\-eligible countries\)|Europe \& Central Asia \(IDA\-eligible countries\)|East Asia \& Pacific \(IDA\-eligible countries\)|South Asia \(IFC classification\)|Middle East and North Africa \(IFC classification\)|Latin America and the Caribbean \(IFC classification\)|Europe and Central Asia \(IFC classification\)|East Asia and the Pacific \(IFC classification\)|Sub\-Saharan Africa \(IFC classification\)|Sub\-Saharan Africa \(IBRD\-only countries\)|Latin America \& the Caribbean \(IBRD\-only countries\)|Middle East \& North Africa \(IBRD\-only countries\)|East Asia \& Pacific \(IBRD\-only countries\)|IBRD countries classified as high income|Europe \& Central Asia \(IBRD\-only countries\)|Sub\-Saharan Africa \(IDA \& IBRD countries\)|Sub-Saharan Africa \(excluding high income\)|Sub\-Saharan Africa|South Asia \(IDA \& IBRD\)|South Asia|Small states|Pre\-demographic dividend|Post\-demographic dividend|Pacific island small states|Other small states|OECD members|Not classified|North America|Middle income|Middle East \& North Africa \(IDA \& IBRD countries\)|Middle East \& North Africa \(excluding high income\)|Middle East \& North Africa|Lower middle income|Low income|Low \& middle income|Least developed countries\: UN classification|Latin America \& the Caribbean \(IDA \& IBRD countries\)|Latin America \& Caribbean \(excluding high income\)|Latin America \& Caribbean|Late\-demographic dividend|IDA total|IDA only|IDA blend|IDA \& IBRD total|IBRD only|High income|Heavily indebted poor countries \(HIPC\)|Fragile and conflict affected situations|European Union|Europe \& Central Asia \(IDA \& IBRD countries\)|Europe \& Central Asia \(excluding high income\)|Europe \& Central Asia|Euro area|East Asia \& Pacific \(IDA \& IBRD countries\)|East Asia \& Pacific \(excluding high income\)|East Asia \& Pacific|Early\-demographic dividend|Central Europe and the Baltics|Caribbean small states|Arab World|Africa Western and Central|Africa Eastern and Southern)$).*$'
        data = wbdata.get_countries(query=expression)
        country_codes=[]
        for item in data:
            if item["id"] != 'DNS' and item["id"] != 'DSF':
                country_codes.append(item["id"])

        # Set the date range
        data_date_range = (datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 12, 31))

        # Fetch data
        print("Fetching data from the World Bank API...")
        data = wbdata.get_dataframe(indicators, country=country_codes, date=data_date_range, parse_dates=True)

        # Reset index and save to CSV
        data.reset_index(inplace=True)
        data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    @staticmethod
    def imputation_country(features, initial_dataset_path, 
                   imputed_training_dataset_path_country, 
                   imputed_testing_dataset_path_country,
                   test_ratio):
        '''
        Docstring for imputation_country

        :param features: features that have to be imputed, only numerical features
        :param initial_dataset_path: path to the data that have to be imputed
        :param imputed_training_dataset_path_country: path to the file there imputed training data for classification should be saved 
        :param imputed_testing_dataset_path_country: path to the file there imputed testing data for classification should be saved
        :param test_ratio: ratio of the data saved for testing
        '''

        # create imputer
        imputer = HybridImputer(path=initial_dataset_path, features=features, split_criteria='country', country_test_size=test_ratio)

        
        # apply imputation
        imputer.imputation()

        # save training data to the file
        imputer.training_data.to_csv(imputed_training_dataset_path_country, index=False)
        print(f"Data saved to {imputed_training_dataset_path_country}")
        # save testing data to the file
        imputer.testing_data.to_csv(imputed_testing_dataset_path_country, index=False)
        print(f"Data saved to {imputed_testing_dataset_path_country}")

    @staticmethod
    def imputation_year(features, initial_dataset_path, 
                        imputed_training_dataset_path_year, 
                        imputed_validation_dataset_path_year, 
                        imputed_testing_dataset_path_year,
                        year_split_sizes):
        
        '''
        Docstring for imputation_year

        :param features: features that have to be imputed, only numerical features
        :param initial_dataset_path: path to the data that have to be imputed
        :param imputed_training_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param imputed_validation_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param imputed_testing_dataset_path_year: path to the file there imputed testing data for time series regression should be saved
        :param year_split_sizes: array containing 2 values, number of years for the validation and number of years for testing 
        '''
        imputer = HybridImputer(path=initial_dataset_path, features=features, split_criteria='year', year_split_sizes=year_split_sizes)
        imputer.imputation()

        imputer.training_data.to_csv(imputed_training_dataset_path_year, index=False)
        print(f"Data saved to {imputed_training_dataset_path_year}")
        imputer.validation_data.to_csv(imputed_validation_dataset_path_year, index=False)
        print(f"Data saved to {imputed_validation_dataset_path_year}")
        imputer.testing_data.to_csv(imputed_testing_dataset_path_year, index=False)
        print(f"Data saved to {imputed_testing_dataset_path_year}")