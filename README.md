# advanced-ml-gdp-forecasting

# Overview 

This project explores machine learning methods for country development classification, GDP forecasting, and data augmentation. An MLP model is used to classify countries into four development levels, while LSTM, CNN–LSTM, and Transformer architectures are evaluated for five-year GDP forecasting based on ten years of historical economic and demographic data. Variational Autoencoders (VAEs) are also investigated as a data augmentation approach to enhance forecasting performance.

This project was completed as part of the Advanced Machine Learning module in the Heriot Watt University.

# Research Objectives

- Fill the missing values in the data using the interpolation and kNN Imputation Techniques.
- Implement MLP and LSTM manually to understand the models and the idea of backpropagation.
- Use MLP model to classify countries into 4 categories based on the development factor
- Compare the performance of LSTM, CNN-LSTM and Transformer models in forecasting performance.
- Investigate the impact of synthetic data generation using Variational Autoencoders (VAE).

# Dataset
The dataset contain the annual country-level economic indicators covering the period from 01/01/1980 to 31/12/2023, including
the following 11 features:
1. Country
2. Date GDP
3. Population
4. Life Expentancy
5. Literacy Rate
6. Unemployment Rate
7. Energy Use
8. Fertility Rate
9. Poverty Ratio
10. Primary School Enrolmet Rate
11. Exports Level

# Data Preprocessing

Several preprocessing techniques were implemented:

Missing Value Imputation
Country-specific interpolation
Linear interpolation
Cubic spline interpolation
Exponentially weighted moving averages
K-Nearest Neighbour imputation
Applied to remaining missing observations
Scaling
StandardScaler fitted exclusively on training data
Validation and test datasets transformed using training statistics


# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

# Author

Fedor Birilov for Statistical Data Science in Heriot-Watt University
