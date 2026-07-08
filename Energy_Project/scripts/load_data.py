import pandas as pd
import os

data_path = 'data/'


print('Step 1: Loading and Exploring Data')


# load smartmeter data 2023
smart_23 = pd.read_csv(os.path.join(data_path, 'sample_23_20percent.csv'))

print(f'\nSmartmeter data 2023:')
print(f'  Shape: {smart_23.shape[0]} rows x {smart_23.shape[1]} columns')
print(f'  First column name: {smart_23.columns[0]}')
print(f'  Number of time series (customers): {smart_23.shape[1] - 1}')
print(f'  First 5 values of first column:\n{smart_23.iloc[:5, 0].values}')

# load weather data
avgtemp_23 = pd.read_csv(os.path.join(data_path, 'sample_avgtemp_23_20percent.csv'))

print(f'\nWeather data (avg temp) 2023:')
print(f'  Shape: {avgtemp_23.shape[0]} rows x {avgtemp_23.shape[1]} columns')
print(f'  First column name: {avgtemp_23.columns[0]}')

# check consistency
print('\nChecking consistency across files:')
files = [
    'sample_23_20percent.csv',
    'sample_avgtemp_23_20percent.csv',
    'sample_maxtemp_23_20percent.csv',
    'sample_mintemp_23_20percent.csv',
    'sample_maxGSun_23_20percent.csv',
    'sample_sumGSun_23_20percent.csv'
]

for f in files:
    df = pd.read_csv(os.path.join(data_path, f))
    print(f'  {f}: {df.shape[0]} rows')


print('Data loading complete')
