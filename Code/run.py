from Solutions.QuestionOneSolutions import QuestionOneSolutions
from Solutions.QuestionTwoSolutions import QuestionTwoSolutions
from Solutions.QuestionThreeSolutions import QuestionThreeSolutions
from Solutions.QuestionFourSolutions import QuestionFourSolutions
from Solutions.QuestionFiveSolutions import QuestionFiveSolutions

def run():

    '''

    Global Parameters that have to be changed:
        :param initial_dataset_path: update as path to initial dataset, this dataset can only contain features from parameter features, otherwise you will get a mistake
        :param imputed_training_dataset_path_country: update to store training data for classification
        :param imputed_testing_dataset_path_country: update to store testing data for classification
        :param imputed_training_dataset_path_year: update to store training data for time series forecast
        :param imputed_validation_dataset_path_year: update to store validation data for time series forecast
        :param imputed_testing_dataset_path_year: update to store testing data for time series forecast

    Global Parameters that can be changed:
        :param INDICATORS: update to indicators you want to download from the worldbank API
        :param FEATURES: update to features you want to use 
        :param RESPONSE_VARIABLE: update to feature that will be use as prediction target
        :param VAL_SIZE: update to number of years for validation
        :param TEST_SIZE: update to number of years for testing
        :param HISTORY_SIZE: update to number of historical years that will be used for prediction
        :param PREDICTION_SIZE: update to number of years you want to predict
    '''
    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE USED INDICATORS TO DOWNLOAD DATA FROM WORLD BANK ───────────────────────────────────────────────────
    INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "GDPpc_2017$", #"GDP per capita, PPP (constant 2017 US$) #original
    "SP.POP.TOTL": "Population_total", #original
    "SP.DYN.LE00.IN": "Life_exectancy" #original
    ,"SE.ADT.LITR.ZS": "Literacy_rate" #original
    ,"SL.UEM.TOTL.ZS": "Unemploymlent_rate" #original
    ##,"EG.USE.PCAP.KG.OE": "Access_electricity" #does not work
    ,"SP.DYN.TFRT.IN": "Fertility_rate" #original
    ,"SI.POV.NAHC": "Poverty_ratio" #original
    ,"SE.PRM.ENRR": "Primary_school_enrolmet_rate" #original
    ,"NE.GDI.TOTL.ZS": "Gross_capital_formation" #added
    ##,"IT.NET.USER.ZS": "Internet_users"
    #,"SE.XPD.TOTL.GD.ZS": "Gov_education_investment"222
    ##,"EN.ATM.CO2E.PC": "CO2_emissions"
    #,"SI.POV.GINI": "Income_inequality"
    #,"SH.H2O.BASW.ZS": "Access_to_water"
    ,"EG.USE.PCAP.KG.OE": "Energy_use" #original
    ##,"NE.EXP.GNFS.CD": "Exports_c$" #Exports of goods and services (current US$)
    ,"NE.EXP.GNFS.KD": "Exports_2017$" #Exports of goods and services (current US$) #original
    #,"SE.XPD.PRIM.PC.ZS":  "Expenditure_primary" #Government expenditure per student, primary (% of GDP per capita)
    #,"SE.XPD.SECO.PC.ZS":  "Expenditure_secodary" #Government expenditure per student, secondary (% of GDP per capita)
    #,"SE.XPD.TERT.PC.ZS":  "Expenditure_tertiary" #"Government expenditure per student, tertiary (% of GDP per capita)"
    }
    # ──────────────────────── UPDATE THIS VARIABLE TO FIRST YEAR OF DATA DOWLOADED FROM WORLD BANK ───────────────────────────────────────────────────
    START_YEAR = 1980
    # ──────────────────────── UPDATE THIS VARIABLE TO LAST YEAR OF DATA DOWLOADED FROM WORLD BANK ───────────────────────────────────────────────────
    END_YEAR = 2023

    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE USED FEATURES IN THE ENTIRE Q3 SOLUTION ───────────────────────────────────────────────────
    FEATURES = ["GDPpc_2017$","Population_total","Life_exectancy",
                "Literacy_rate","Unemploymlent_rate","Energy_use",
                "Fertility_rate","Poverty_ratio","Primary_school_enrolmet_rate",
                "Gross_capital_formation","Exports_2017$"]
    # ──────────────────────── Number of historical years used for prediciton ───────────────────────────────────────────────────
    HISTORY_SIZE = 10
    # ──────────────────────── Number of years to predict ───────────────────────────────────────────────────
    PREDICTION_SIZE = 5
    # ──────────────────────── Prediction Target variable name ───────────────────────────────────────────────────
    RESPONSE_VARIABLE = "GDPpc_2017$"

    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO INITIAL DATASET ───────────────────────────────────────────────────
    initial_dataset_path = '/Users/arina/Documents/uni/year 4/Adv_ML/final_data.csv'

    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO STORE TRAINING CLASSIFICATION DATASET ───────────────────────────────────────────────────
    imputed_training_dataset_path_country = '/Users/arina/Documents/uni/year 4/Adv_ML/DataStorage/imputed_training_dataset_country.csv'
    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO STORE TESING CLASSIFICATION DATASET ───────────────────────────────────────────────────
    imputed_testing_dataset_path_country = '/Users/arina/Documents/uni/year 4/Adv_ML/DataStorage/imputed_testing_dataset_country.csv'

    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO STORE TRAINING TIMSERIES DATASET ───────────────────────────────────────────────────
    imputed_training_dataset_path_year = '/Users/arina/Documents/uni/year 4/Adv_ML/DataStorage/imputed_training_dataset_year.csv'
    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO STORE VALIDATION TIMSERIES DATASET ───────────────────────────────────────────────────
    imputed_validation_dataset_path_year = '/Users/arina/Documents/uni/year 4/Adv_ML/DataStorage/imputed_validation_dataset_year.csv'
    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH TO STORE TESING TIMSERIES DATASET ───────────────────────────────────────────────────
    imputed_testing_dataset_path_year = '/Users/arina/Documents/uni/year 4/Adv_ML/DataStorage/imputed_testing_dataset_year.csv'
    # ──────────────────────── UPDATE THIS VARIABLE TO UPDATE PATH FOR FOLDER TO STORE ALL PLOTS ───────────────────────────────────────────────────
    PLOTS_FOLDER_PATH = ('/Users/arina/Documents/uni/year 4/Adv_ML/EDA')
    
    # ──────────────────────── NUMBER OF YEARS FOR VALIDATION SUBSET ───────────────────────────────────────────────────
    VAL_SIZE = 10
    # ──────────────────────── NUMBER OF YEARS FOR TESTING SUBSET ───────────────────────────────────────────────────
    TEST_SIZE = 5
    

    ######################################################################################################
    # ──────────────────────── SOLUTION FOR QUESTION 1 ───────────────────────────────────────────────────
    ######################################################################################################

    '''Q_one = QuestionOneSolutions(initial_dataset_path=initial_dataset_path,
                                 imputed_training_dataset_path_country=imputed_training_dataset_path_country,
                                 imputed_testing_dataset_path_country=imputed_testing_dataset_path_country,
                                 imputed_training_dataset_path_year=imputed_training_dataset_path_year,
                                 imputed_validation_dataset_path_year=imputed_validation_dataset_path_year,
                                 imputed_testing_dataset_path_year=imputed_testing_dataset_path_year,
                                 plots_folder_path=PLOTS_FOLDER_PATH, 
                                 indicators=INDICATORS,
                                 features=FEATURES, 
                                 start_year=START_YEAR,
                                 end_year=END_YEAR)
    dataset = Q_one.solve_question()'''

    ######################################################################################################
    # ────────────────────────────────────────────────────────────────────────────────────────────────────
    ######################################################################################################
    

    ######################################################################################################
    # ──────────────────────── SOLUTION FOR QUESTION 2 ───────────────────────────────────────────────────
    ######################################################################################################
    
    ''' Q_two = QuestionTwoSolutions(dataset=dataset, features=FEATURES)
    Q_two.solve_question()'''

    ######################################################################################################
    # ────────────────────────────────────────────────────────────────────────────────────────────────────
    ######################################################################################################


    ######################################################################################################
    # ──────────────────────── SOLUTION FOR QUESTION 3 ───────────────────────────────────────────────────
    ######################################################################################################

    '''Q_three = QuestionThreeSolutions(initial_dataset_path=initial_dataset_path,
                                 imputed_training_dataset_path_year=imputed_training_dataset_path_year,
                                 imputed_validation_dataset_path_year=imputed_validation_dataset_path_year,
                                 imputed_testing_dataset_path_year=imputed_testing_dataset_path_year,
                                 features=FEATURES,
                                 response_variable=RESPONSE_VARIABLE,
                                 prediction_size=PREDICTION_SIZE,
                                 history_size=HISTORY_SIZE)
    Q_three.solve_question()'''

    ######################################################################################################
    # ────────────────────────────────────────────────────────────────────────────────────────────────────
    ######################################################################################################



    ######################################################################################################
    # ──────────────────────── SOLUTION FOR QUESTION 4 ───────────────────────────────────────────────────
    ######################################################################################################

    '''Q_four = QuestionFourSolutions(imputed_training_dataset_path_year=imputed_training_dataset_path_year,
                                 imputed_validation_dataset_path_year=imputed_validation_dataset_path_year,
                                 imputed_testing_dataset_path_year=imputed_testing_dataset_path_year,
                                 features=FEATURES,
                                 response_variable=RESPONSE_VARIABLE,
                                 prediction_size=PREDICTION_SIZE,
                                 history_size=HISTORY_SIZE)
    Q_four.solve_question()'''

    ######################################################################################################
    # ────────────────────────────────────────────────────────────────────────────────────────────────────
    ######################################################################################################

    ######################################################################################################
    # ──────────────────────── SOLUTION FOR QUESTION 5 ───────────────────────────────────────────────────
    ######################################################################################################

    '''Q_five = QuestionFiveSolutions(imputed_training_dataset_path_year=imputed_training_dataset_path_year,
                                 imputed_validation_dataset_path_year=imputed_validation_dataset_path_year,
                                 imputed_testing_dataset_path_year=imputed_testing_dataset_path_year,
                                 features=FEATURES,
                                 response_variable=RESPONSE_VARIABLE,
                                 prediction_size=PREDICTION_SIZE,
                                 history_size=HISTORY_SIZE)
    Q_five.solve_question()'''

    ######################################################################################################
    # ────────────────────────────────────────────────────────────────────────────────────────────────────
    ######################################################################################################

if __name__ == '__main__':
    run()
