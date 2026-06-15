from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(path, sep):
    # function to read the data from the csv file

    dataset =  pd.read_csv(path, sep=sep)
    return dataset.sort_values(['country', 'date']).reset_index(drop=True)

class HybridImputer:

    '''
    Docstring for HybridImputer
    Hybrid method for the data imputation, starting with country first, and then global imputation techniques
    '''

    def __init__(self, path, features, split_criteria, country_column = "country", country_test_size = 0.2, year_split_sizes = [5, 5]):

        '''
        Docstring for __init__
        
        :param self: Description
        :param path: the path to the dataframe with potential missing values
        :param features: should be initialized with the array of numerical features that should be potentionally considered for data imputation 
        :param country_column: name of a column used to access country column in the initial dataframe
        :param split_criteria: this parameter is used to make decision regarding the train test split to avoi data leakage, use split_criteria = country for classification and split_criteria = year for time sereis regression 
        :param country_test_size: the ratio of the testing data out of whole dataset for the country split only
        :param year_split_sizes: array with 2 parameters, there 1 value is number of years for validation, and 2nd is number of years for testing, used for year split
        '''

        self.data = read_csv(path=path, sep=",")
        self.country_col = country_column
        self.features = features
        self.countries = self.data[self.country_col].unique()
        self.imputation_log = []
        self.country_names = self.get_country_names()
        self.split_criteria = split_criteria
        self.country_test_size = country_test_size
        self.year_split_size = year_split_sizes
        
    def get_country_names(self):
        # get unique list of countries from the raw dataframe

        return self.data['country'].unique()

    @staticmethod
    def data_checks(data):
        '''
        Docstring for data_checks

        :param data: dataframe with the data that has to be analysed
        Checks for missing and duplicate values, along with the different types of variables 
        that are contained in the data provided
        '''
        
        analysis = []

        for feature in data.columns:
            analysis.append({
                "column_name" : feature,
                "num_miss_values": data[feature].isnull().sum(),
                "percent_miss_values": round(data[feature].isna().mean() * 100, 1),
                "data_type": data[feature].dtype
            })
        print(pd.DataFrame(analysis))

        print(54*'#')
        print("TOTAL NUMBER OF MISSING VALUES FOR NUMERIC COLUMNS: ", data.select_dtypes(include='number').isnull().sum().sum())
        print("TOTAL PERCENT OF MISSING VALUES FOR NUMERIC COLUMNS: ", round(data.select_dtypes(include='number').isnull().sum().sum()/data.select_dtypes(include='number').size*100, 1), "%")
        print("DIMENTIONS OF THE NUMERICAL DATA", data.select_dtypes(include='number').shape)
        print("NUMBER OF COUNTIRIES IN THE DATA", data['country'].nunique())
    
    @staticmethod
    def missing_values_for_certain_countires(data, countries_names):
        '''
        Docstring for missing_values_for_certain_countires

        :param data: dataframe with the data that has to be analysed
        :param countries_names: array with names of countries that have to be checked

        Checks for missing , along with the different types of variables 
        that are contained in the data provided for a certain countries
        '''

        for country in countries_names:
            country_data = data[data["country"] == country]
            print(f"TOTAL NUMBER OF MISSING VALUES FOR NUMERIC COLUMNS FOR {country}: ", country_data.select_dtypes(include='number').isnull().sum().sum())
            print(f"TOTAL PERCENT OF MISSING VALUES FOR NUMERIC COLUMNS FOR {country}: ", round(country_data.select_dtypes(include='number').isnull().sum().sum()/country_data.select_dtypes(include='number').size*100, 1), "%")
 
    def get_countries_data(self):
        # get an array, where each element contain a dataframe with data from a certain country

        country_array = []
        for country in self.country_names:

            country_array.append(self.data[self.data['country'] == country])

        return country_array

    def apply_train_test_split(self, random_state = 42):

        '''
        Docstring for apply_train_test_split

        This function is making the test train split on a country basis

        :param test_size: parameter is used to define precentage of dataset that have to be used for testing, default is 20% of data
        :param random_state: parameter is used to keep consistency is data split, default is 42

        '''

        if self.split_criteria == "country":
            country_array = self.get_countries_data()

            train, test = train_test_split(country_array, test_size=self.country_test_size, random_state=random_state, shuffle=True)
            return pd.concat(train, ignore_index=True), None, pd.concat(test, ignore_index=True)
        
        if self.split_criteria == "year":
            train = pd.concat([
            group.sort_values("date").iloc[:-(self.year_split_size[0]+self.year_split_size[1])]
            for country, group in self.data.groupby("country")
            ], ignore_index=True)

            validation = pd.concat([
                group.sort_values("date").iloc[-(self.year_split_size[0]+self.year_split_size[1]):-self.year_split_size[1]]
                for country, group in self.data.groupby("country")
            ], ignore_index=True)

            test = pd.concat([
                group.sort_values("date").iloc[-self.year_split_size[1]:]
                for country, group in self.data.groupby("country")
            ], ignore_index=True)
            return train, validation, test
        
    def get_max_gap(self, country, feature):

        ''' 
        Docstring for get_max_gap

        function to check the maximum number of cosequtive gaps in the series
        used to make a decision regarding the method that is used for data imputation

        '''

        # take the series of the data from the selected country of a certain feature
        series = self.data.loc[self.data[self.country_col] == country, feature]
        # check each values in the sequence if it Nan or not, returns the True/ False
        is_nan = series.isnull().values

        current_gap = 0
        max_gap = 0

        # max_gap will contain the longest consequtive gap in the dataset
        for val in is_nan:
            if val:
                current_gap+=1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        return max_gap

    def select_and_apply_imputation_method(self, country, feature):

        '''
        Docstring for select_and_apply_imputation_method

        this function if used to select an appropriate imputation method, 
        based on the lenght of a max consequtive gap in the series:

        max_gap == 0: do not touch this feature for this country
        max_gap == 1: implement linear interpolation
        2 <= max_gap <= 5: implement cubic spline interpolation (solve cubic polinomial)
        6 <= max_gap <= 15: implement linear interpolation
        16 <= max_gap: use global dataset with kNNImputer from sclearn
        '''

        method = ""
        max_gap = self.get_max_gap(country=country, feature=feature)

        if max_gap == 0:
            #print(f"{country} has no missing values for feature {feature}")
            method = "NO_IMPUTATION"

        elif max_gap == 1:
            #print(f"{country} has max_gap = 1 for feature {feature}")
            self.apply_interpolation(country=country, feature=feature, method='linear', limit=1)
            method = "LINEAR_INTERPOLATION"
                
        elif 2 <= max_gap <= 5:
            #print(f"{country} has 2 <= max_gap <= 5 for feature {feature}")
            self.apply_interpolation(country=country, feature=feature, method='spline', limit=5, order=3)
            method = "CUBIC_SPLINE_INTERPOLATION"
                
        elif 6 <= max_gap <= 15:
            #print(f"{country} has 6 <= max_gap <= 15 for feature {feature}")
            self.apply_interpolation(country=country, feature=feature, method='linear', limit=15)
            method = "LINEAR_INTERPOLATION"

        elif 16 <= max_gap:
            #print(f"{country} has 16 <= max_gap for feature {feature}")
            method = "kNN_IMPUTER"

        self.imputation_log.append({
                    'country': country,
                    'feature': feature,
                    'Method': method
                })

    def apply_interpolation(self, country, feature, limit, method, order=None):

        '''
        Docstring for apply_interpolation
        Application of interpolation for a certain variable and country
        
        :param self: Description
        :param country: Country you need the interpolation for 
        :param feature: Variable you need the interpolation for 
        :param limit: The maximum value of consequtive values that can be interpolated
        :param method: The method of interpolation ('linear' or 'spline' for example)
        :param order: order of polinomial, this variable is only used for spline interpolation
        '''

        series = self.data.loc[self.data[self.country_col] == country, feature]
        if self.split_criteria == "year":
            # forward interpolation to avoid data leakage in time series
            self.data.loc[self.data[self.country_col] == country, feature] = series.interpolate(method=method, limit_direction='forward', limit=limit, order=order)
        else:
            self.data.loc[self.data[self.country_col] == country, feature] = series.interpolate(method=method, limit_direction='both', limit=limit, order=order)

    def apply_kNN_imputer(self, n_neighbors = 9):


        '''
        function is used for kNN_Imputer application, with prior data scaling
        and reserving back to original scaling
        '''

        # split the data to the test and training split, to plrevent the data leakage coming from kNN usage
        self.training_data, self.validation_data, self.testing_data = self.apply_train_test_split()

        # create a Standart scaler object
        scaler = StandardScaler()
        # learn the mean and sd from the training data and transform the data
        scaled_training_data = scaler.fit_transform(self.training_data[self.features].copy())
        # use scaler from training data to transform the testing data, thi sprevents data leakage
        scaled_testing_data = scaler.transform(self.testing_data[self.features].copy())

        imputer = KNNImputer(n_neighbors=n_neighbors)
        # train the kNN imputer on the training data and generate missing values
        imputed_training_data =imputer.fit_transform(scaled_training_data)
        # use kNN imputer trained on training data to generate missing values for testing data
        imputed_testing_data = imputer.transform(scaled_testing_data)

        # get training data to its original scaling
        inversed_training_data = scaler.inverse_transform(imputed_training_data)
        # get testing data to its original scaling
        inversed_testing_data = scaler.inverse_transform(imputed_testing_data)

        imputed_training_df = pd.DataFrame(inversed_training_data, columns=self.features, index=self.training_data.index)
        imputed_testing_df = pd.DataFrame(inversed_testing_data, columns=self.features, index=self.testing_data.index)

        # fill missing values in training dataset
        self.training_data = self.training_data.fillna(imputed_training_df)
        # fill missing values in testing dataset
        self.testing_data = self.testing_data.fillna(imputed_testing_df)
        # in case if validation data is required
        if self.split_criteria == "year":
            scaled_validation_data = scaler.transform(self.validation_data[self.features].copy())
            imputed_validation_data =imputer.transform(scaled_validation_data)
            inversed_validation_data =scaler.inverse_transform(imputed_validation_data)
            imputed_validation_df = pd.DataFrame(inversed_validation_data, columns=self.features, index=self.validation_data.index)
            self.validation_data = self.validation_data.fillna(imputed_validation_df)

    def imputation(self):
        
        '''
        Docstring for imputation

        Global function for imputation. First 2 for loops implement simple statistical imputation
        methods on a country basis. The next set of for loops are to implement global kNN imputer
        from sklearn for those variables that have to much missing data

        This function can be called after initialization to perform full data imputation
        Imputed dataset will availible in self.data variable and can be accesed
        after imputation() is called
        '''
            
        # Imputation on a country level for features with less than 16 consequtive missing values
        for country in self.countries:
            for feature in self.features:
                self.select_and_apply_imputation_method(country=country, feature=feature)

        # Imputation on a global level for features with more than 15 missing values
        self.apply_kNN_imputer()

        print("############################---HYBRID IMPUTER HAS BEEN EXECUTED WITH NO MISTAKES---################################################")
        print(54*'#', 'CHECKS FOR TRAINING DATA', 54*'#')
        HybridImputer.data_checks(self.training_data)
        if self.split_criteria == "year":
            print(54*'#', 'CHECKS FOR VALIDATION DATA', 54*'#')
            HybridImputer.data_checks(self.validation_data)
        print(54*'#', 'CHECKS FOR TESTING DATA', 54*'#')
        HybridImputer.data_checks(self.testing_data)
