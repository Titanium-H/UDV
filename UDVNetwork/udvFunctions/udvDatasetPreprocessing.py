# Name: Loading pre-processing (Regression)
# Function: load and pre-process dataset(s)
#===========================================================
# Necessary package
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import datetime
#===========================================================
#==========================START============================
# Pre-process dataset: New York City Taxi Trip Duration
# Meg Risdal. (2017). New York City Taxi Trip Duration. Kaggle. https://kaggle.com/competitions/nyc-taxi-trip-duration
# Train file will be split into training and validation

class NYCDataset(Dataset): 
    def __init__(self, data_path):
        # Load original data from data_path
        self.data = pd.read_csv(data_path)
        self.features = self.data.iloc[:, 1:-1].values      # input features: NO first column (unique ID) or last column (duration)
        self.targets = self.data.iloc[:, -1].values         # output feature: last column (duration)
        self.features = np.delete(self.features, 2, axis=1) # Delete drop time ,if not, the duration can be deduced by pickup time and drop time
        
        # Convert continuous time variables to discrete time variables
        self.time_year = []
        self.time_month = []
        self.time_day = []
        self.time_hour = []
        self.time_minute = []
        self.time_second = []
        
        for ind in range(0, len(self.features[:,1])):
            time_obj = datetime.datetime.strptime(self.features[ind, 1], '%Y-%m-%d %H:%M:%S')
            self.time_year.append(time_obj.year)               # Year is discarded since all data comes from the same year
            self.time_month.append(time_obj.month)             # Month
            self.time_day.append(time_obj.day)                 # Day
            self.time_hour.append(time_obj.hour)               # Hour
            self.time_minute.append(time_obj.minute)           # Minute
            self.time_second.append(time_obj.second)           # Second
        self.features = np.delete(self.features, 1, axis = 1)  # Discard continuous time
        
        self.features = np.insert(self.features, 1, self.time_month, axis = 1)
        self.features = np.insert(self.features, 2, self.time_day, axis = 1)
        self.features = np.insert(self.features, 3, self.time_hour, axis = 1)
        self.features = np.insert(self.features, 4, self.time_minute, axis = 1)
        self.features = np.insert(self.features, 5, self.time_second, axis = 1)

        # Convert string data to number
        self.string = []
        for col in range (0, len(self.features[0, :])):
            if type(self.features[0, col]) == str:
                self.string.append(col)
            else:
                for row in range (0, len(self.features[:, 0])):
                    if type(self.features[row, col]) == str:
                        self.string.append(col)
                        break
        
        str2num = LabelEncoder()
        for i in range (0, len(self.string)):
            self.features[:, self.string[i]] = str2num.fit_transform(self.features[:, self.string[i]])
        
        # Normalisation: [0,1]
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = feature_scaler.fit_transform(self.features)
        self.targets = target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # Fill the missing items
        imputer = SimpleImputer(strategy='mean')
        self.features = imputer.fit_transform(self.features)
        self.targets = imputer.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # Convert to float32
        self.features = self.features.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        target = self.targets[idx]
        return sample, target
    
#===========================END=============================

#==========================START============================
# Pre-process dataset: House Prices - Advanced Regression Techniques
# Anna Montoya, DataCanary. (2016). House Prices - Advanced Regression Techniques. Kaggle. https://kaggle.com/competitions/house-prices-advanced-regression-techniques
# Train file be split into training and validation

class HousingPriceDataset(Dataset):
    def __init__(self, data_path):
        # Load original data from data_path
        self.data = pd.read_csv(data_path)
        self.features = self.data.iloc[:, 1:-1].values # Input features: NO first column (unique ID) Or last column (price)
        self.targets = self.data.iloc[:, -1].values    # Output feature: Price
        
        # Convert string data to number
        self.string = []
        for col in range (0, len(self.features[0, :])):
            if type(self.features[0, col]) == str:
                self.string.append(col)
            else:
                for row in range (0, len(self.features[:, 0])):
                    if type(self.features[row, col]) == str:
                        self.string.append(col)
                        break
        
        str2num = LabelEncoder()
        for i in range (0, len(self.string)):
            self.features[:, self.string[i]] = str2num.fit_transform(self.features[:, self.string[i]])
        
        # Normalisation: [0,1]
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = feature_scaler.fit_transform(self.features)
        self.targets = target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # Fill the missing items
        imputer = SimpleImputer(strategy='mean')
        self.features = imputer.fit_transform(self.features)
        self.targets = imputer.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # Convert to float32
        self.features = self.features.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        target = self.targets[idx]
        return sample, target
#===========================END=============================