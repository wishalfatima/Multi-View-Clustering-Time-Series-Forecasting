import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import STL

def normalize_dataset(dataset, look_forward):
    """
    Normalize dataset and extract mean values.
    
    Args:
        dataset: List of time series (variable lengths)
        look_forward: Forecast horizon
    
    Returns:
        normalized_dataset: List of normalized series
        data_means: List of means for each series
    """
    normalized_dataset = []
    data_means = []
    
    for ts in dataset:
        mean_val = np.mean(ts)
        data_means.append(mean_val)
        
        if mean_val != 0:
            normalized_dataset.append(ts / mean_val)
        else:
            normalized_dataset.append(ts)
    
    return np.array(normalized_dataset, dtype=object), np.array(data_means)


def rescale_data_to_main_value(data, data_means, dataset_seasonal):
    """
    Rescale data back to original scale using means and seasonal components.
    """
    print(f"DEBUG - data shape: {data.shape}")
    print(f"DEBUG - data_means shape: {data_means.shape}")
    print(f"DEBUG - dataset_seasonal shape: {dataset_seasonal.shape if hasattr(dataset_seasonal, 'shape') else 'not an array'}")
    
    # Check if dataset_seasonal exists and has elements
    if dataset_seasonal is not None and len(dataset_seasonal) > 0:
        # If it's a numpy array, we need to handle it differently
        if isinstance(dataset_seasonal, np.ndarray):
            # Reshape data_means to match data dimensions
            if len(data.shape) > 1:
                # Reshape data_means from (145,) to (145, 1)
                data_means_reshaped = data_means.reshape(-1, 1)
                print(f"DEBUG - data_means_reshaped shape: {data_means_reshaped.shape}")
                
                # Try to reshape seasonal
                if len(dataset_seasonal.shape) == 1:
                    seasonal_reshaped = dataset_seasonal.reshape(-1, 1)
                    print(f"DEBUG - seasonal_reshaped shape: {seasonal_reshaped.shape}")
                    return (data * data_means_reshaped) + seasonal_reshaped
                else:
                    return (data * data_means_reshaped) + dataset_seasonal
            else:
                return (data * data_means) + dataset_seasonal
        else:
            return data * data_means
    else:
        return data * data_means

def normalize_feature_vectors(features):
    """Applies Min-Max Normalization to feature vectors."""
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)

    features = (features - minimum) / (maximum - minimum + 1e-8)  # Avoid division by zero
    return features

# Dataset Creation for Time-Series Forecasting
def create_dataset(sample, look_back, look_forward, sample_overlap, dataset_seasonal):
    """Creates sliding window samples for training."""
    if sample_overlap >= look_forward or sample_overlap < 0:
        sample_overlap = look_forward - 1
    if look_forward == 1:
        sample_overlap = 0

    dataX, dataY, dataY_seasonal = [], [], []
    
    for i in range(0, len(sample) - look_back - look_forward + 1, look_forward - sample_overlap):
        dataX.append(sample[i:(i + look_back), 0])
        dataY.append(sample[(i + look_back):(i + look_back + look_forward), 0])
        dataY_seasonal.append(dataset_seasonal[(i + look_back):(i + look_back + look_forward)])

    return np.array(dataX), np.array(dataY), np.array(dataY_seasonal)

def create_dataset2(sample, look_back, look_forward, sample_overlap, dataset_seasonal, dataset_name):
    """Creates dataset with trend & seasonal forecasting using Exponential Smoothing."""
    if sample_overlap >= look_forward or sample_overlap < 0:
        sample_overlap = look_forward - 1
    if look_forward == 1:
        sample_overlap = 0

    sample_trn = sample[:-look_forward]
    frequency = {'tourism': 4, 'cif-6': 12}.get(dataset_name, 12)

    # Handle cases with shorter sequences
    if len(sample_trn) > 2 * frequency:
        model = ExponentialSmoothing(sample_trn.flatten(), seasonal_periods=frequency, trend='add', seasonal='add').fit()
    elif len(sample_trn) > frequency:
        model = ExponentialSmoothing(sample_trn.flatten(), seasonal_periods=frequency // 2, trend='add', seasonal='add').fit()
    else:
        model = ExponentialSmoothing(sample_trn.flatten()).fit()

    forecast = model.forecast(steps=look_forward).reshape(1, -1)

    # Combine training & forecasted values
    augmented_series = np.concatenate([sample_trn[-(look_back + look_forward - 1):].reshape(-1, 1), forecast.T])
    augmented_series = augmented_series.flatten()

    aug_trainX, aug_trainY = [], []

    for i in range(0, len(augmented_series) - look_back - look_forward + 1, look_forward - sample_overlap):
        aug_trainX.append(augmented_series[i:(i + look_back)])
        aug_trainY.append(augmented_series[(i + look_back):(i + look_back + look_forward)])

    return np.array(aug_trainX), np.array(aug_trainY)


#def stl_decomposition(dataset, frequency, look_forward):
 #   """
  #  Perform STL decomposition on time series data.
    
   # Args:
    #    dataset: List of time series
     #   frequency: Seasonal period (must be odd integer >= 3)
      #  look_forward: Forecast horizon
    
   # Returns:
    #    dataset: Original dataset
     #   seasonal: Seasonal components
      #  trend: Trend components
    #"""
    #seasonal = []
    #trend = []
    
    # Ensure frequency is odd and >= 3 for STL
    #seasonal_period = frequency
    #if seasonal_period < 3:
     #   seasonal_period = 3
    #if seasonal_period % 2 == 0:
     #   seasonal_period = seasonal_period + 1  # Make it odd
      #  print(f"Adjusted seasonal period from {frequency} to {seasonal_period} for STL")
    
    #for ts in dataset:
        # Need at least 2 * seasonal_period data points for decomposition
     #   if len(ts) > 2 * seasonal_period:
      #      try:
       #         stl = STL(ts, period=seasonal_period, seasonal='periodic').fit()
        #        seasonal.append(stl.seasonal)
         #       trend.append(stl.trend)
          #  except Exception as e:
           ##
           #print(f"STL decomposition failed: {e}. Using zeros.")
            #    seasonal.append(np.zeros_like(ts))
             #   trend.append(np.zeros_like(ts))
        #else:
            # Not enough data for decomposition
         #   seasonal.append(np.zeros_like(ts))
          #  trend.append(np.zeros_like(ts))
    
   # return dataset, np.array(seasonal, dtype=object), np.array(trend, dtype=object)
   

def stl_decomposition(dataset, frequency, look_forward):
    """
    Perform STL decomposition on time series data.
    
    Args:
        dataset: List of time series (can have variable lengths)
        frequency: Seasonal period (must be odd integer >= 3)
        look_forward: Forecast horizon
    
    Returns:
        dataset: Original dataset
        seasonal: Seasonal components
        trend: Trend components
    """
    seasonal = []
    trend = []
    
    # Ensure frequency is odd and >= 3 for STL
    seasonal_period = frequency
    if seasonal_period < 3:
        seasonal_period = 3
    if seasonal_period % 2 == 0:
        seasonal_period = seasonal_period + 1  # Make it odd
        print(f"Adjusted seasonal period from {frequency} to {seasonal_period} for STL")
    
    for i, ts in enumerate(dataset):
        # Need at least 2 * seasonal_period data points for decomposition
        if len(ts) > 2 * seasonal_period:
            try:
                stl = STL(ts, period=seasonal_period, seasonal='periodic').fit()
                seasonal.append(stl.seasonal)
                trend.append(stl.trend)
            except Exception as e:
                print(f"STL decomposition failed for series {i}: {e}. Using zeros.")
                seasonal.append(np.zeros_like(ts))
                trend.append(np.zeros_like(ts))
        else:
            # Not enough data for decomposition
            print(f"Series {i} too short (len={len(ts)}) for STL. Using zeros.")
            seasonal.append(np.zeros_like(ts))
            trend.append(np.zeros_like(ts))
    
    # Return as object arrays to handle variable lengths
    return dataset, np.array(seasonal, dtype=object), np.array(trend, dtype=object)


def create_sample(
    look_forward: int,
    sample_seasonal: np.ndarray,
    dataX: np.ndarray, 
    dataY: np.ndarray, 
    data_mean: float, 
    dataY_seasonal: np.ndarray, 
    frequency: int
):
    """
    Creates training, validation, and test sets for time series forecasting.

    Args:
        look_forward (int): Forecasting horizon.
        sample_seasonal (np.ndarray): Seasonal component of the dataset.
        dataX (np.ndarray): Input features.
        dataY (np.ndarray): Target values.
        data_mean (float): Mean of the dataset.
        dataY_seasonal (np.ndarray): Seasonal values of target.
        frequency (int): Seasonal period.

    Returns:
        tuple: Train, validation, and test datasets along with seasonal components.
    """
    test_size, val_size = 1, 1
    train_size = len(dataX) - test_size
    train_size0 = train_size - look_forward + 1

    trainX, testX = dataX[:train_size0-val_size], dataX[train_size:]
    trainY, testY = dataY[:train_size0-val_size], dataY[train_size:]

    valX, valY = dataX[train_size0-val_size:train_size0], dataY[train_size0-val_size:train_size0]

    trainX, valX, testX = map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [trainX, valX, testX])

    val_means = np.full(len(valY), data_mean)
    test_means = np.full(len(testY), data_mean)
    
    val_seasonal = dataY_seasonal[train_size0-val_size:train_size0]
    
    sample_size = len(sample_seasonal.flatten()) - look_forward
    train3 = sample_seasonal[:sample_size].flatten()

    if frequency:
        sp = frequency if len(train3) > 2 * frequency else max(1, frequency // 2)
        fit = ExponentialSmoothing(pd.Series(train3), seasonal_periods=sp, trend='add', seasonal='add').fit()
        preds2 = fit.forecast(steps=look_forward).values.reshape(1, -1)
    else:
        preds2 = np.zeros((1, look_forward))

    test_seasonal_y = dataY_seasonal[train_size:]

    return (trainX, valX, testX, trainY, valY, testY, 
            test_means, val_means, val_seasonal, test_seasonal_y, preds2)
    



def preprocess_dataset(
    all_dataset: list, 
    lag: int, 
    look_forward: int, 
    sample_overlap: int, 
    data_means: list, 
    dataset_seasonal: list,
    frequency: int
):
    """
    Prepares the dataset for training, validation, and testing by processing multiple time series.

    Args:
        all_dataset (list): List of time series datasets.
        lag (int): Window size for input sequences.
        look_forward (int): Forecasting horizon.
        sample_overlap (int): Overlapping samples.
        data_means (list): Mean values for each dataset.
        dataset_seasonal (list): Seasonal components for each dataset.
        frequency (int): Seasonal frequency.

    Returns:
        tuple: Processed train, validation, and test datasets.
    """
    trainX, trainY, valX, valY, testX, testY = [], [], [], [], [], []
    all_test_means, all_val_means, all_test_seasonals, all_test_seasonals2, all_val_seasonals = [], [], [], [], []

    for index, series in enumerate(all_dataset):
        sample = np.array(series).reshape(-1, 1)
        dataX_s, dataY_s, dataY_seasonal = create_dataset(sample, lag, look_forward, sample_overlap, dataset_seasonal[index])

        temp_data = create_sample(look_forward, dataset_seasonal[index], dataX_s, dataY_s, data_means[index], dataY_seasonal, frequency)

        temp_trainX, temp_valX, temp_testX, temp_trainY, temp_valY, temp_testY, test_means, val_means, val_seasonal, test_seasonal, preds2 = temp_data
        
        trainX.extend(temp_trainX.tolist())
        trainY.extend(temp_trainY.tolist())
        valX.extend(temp_valX.tolist())
        valY.extend(temp_valY.tolist())
        testX.extend(temp_testX.tolist())
        testY.extend(temp_testY.tolist())

        all_test_means.extend(test_means.tolist())
        all_val_means.extend(val_means.tolist())
        all_test_seasonals.extend(test_seasonal.tolist())
        all_test_seasonals2.extend(preds2.tolist())
        all_val_seasonals.extend(val_seasonal.tolist())

    return (np.array(trainX), np.array(valX), np.array(testX),
            np.array(trainY), np.array(valY), np.array(testY),
            np.array(all_test_means), np.array(all_val_means),
            np.array(all_val_seasonals), np.array(all_test_seasonals), 
            np.array(all_test_seasonals2))

def save_prediction_result(data: np.ndarray, dataset_name: str = 'dataset', dataset_path: str = ''):
    """
    Saves the prediction results to a CSV file.

    Args:
        data (np.ndarray): Predicted values.
        dataset_name (str): Name of the dataset.
        dataset_path (str): Path to save the CSV file.
    """
    filename = f"{dataset_path}/{dataset_name}-results.csv" if dataset_name else "results.csv"
    pd.DataFrame(data).to_csv(filename, index=False, header=False)
