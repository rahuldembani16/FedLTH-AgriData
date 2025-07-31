import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset/crop_yield_10k_records.csv')

print('Dataset shape:', df.shape)
print('\nColumn info:')
print(df.info())
print('\nFirst few rows:')
print(df.head())
print('\nBasic statistics:')
print(df.describe())
print('\nUnique values per categorical column:')
for col in ['Region', 'Crop_Type', 'Soil_Type']:
    print(f'{col}: {df[col].nunique()} unique values - {df[col].unique()}')

print('\nTarget variable distribution:')
print(df['Yield_tons_per_hectare'].describe())
