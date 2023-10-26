import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset


class GapFillingDataset(Dataset):
    def __init__(self, data_paths, feature_list):
        self.feature_list = feature_list
        self.data_paths = data_paths
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file, key = self.data_paths[idx]
        
        # Load target and covariate data from the HDF5 file
        with h5py.File(file, 'r') as h5file:
            target_data = pd.DataFrame(h5file[key]['target'])
            covariate_data = pd.DataFrame(h5file[key]['covariates'])
            unix_date = h5file[key]['target'][:,0]

        # Assign column names
        target_data.columns = ['date', 'avg_target', 'target', 'mask']
        covariate_data.columns = ['date'] + self.feature_list

        # Convert Unix timestamps to pandas datetime objects
        target_data['date'] = pd.to_datetime(target_data['date'], unit='s')
        covariate_data['date'] = pd.to_datetime(covariate_data['date'], unit='s')

        # Extract the features and the target
        covariates = covariate_data[self.feature_list].values
        avg_target = target_data['avg_target'].values
        target = target_data['target'].values
        mask = target_data['mask'].values
        
        # Extract time features
        hour_of_day = target_data['date'].dt.hour
        day_of_week = target_data['date'].dt.dayofweek # Monday=0, ..., Sunday=6
        day_of_month = target_data['date'].dt.day - 1 # 0-indexed
        day_of_year = target_data['date'].dt.dayofyear - 1 # 0-indexed

        # Calculate number of days
        days_in_month = target_data['date'].dt.days_in_month
        year = target_data['date'].dt.year
        is_leap_year = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)))
        days_in_year = is_leap_year.where(is_leap_year, 365).where(~is_leap_year, 366)

        # Calculate hourly progressions
        hour_of_week = (day_of_week * 24 + hour_of_day) / (7 * 24)
        hour_of_month = (day_of_month * 24 + hour_of_day) / (days_in_month * 24)
        hour_of_year = (day_of_year * 24 + hour_of_day) / (days_in_year * 24)

        # Convert to tensors
        covariates = torch.tensor(covariates, dtype=torch.float32)
        avg_target = torch.tensor(avg_target, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(-1)
        hour_of_day = torch.tensor(hour_of_day, dtype=torch.float32).unsqueeze(-1)
        hour_of_week = torch.tensor(hour_of_week, dtype=torch.float32).unsqueeze(-1)
        hour_of_month = torch.tensor(hour_of_month, dtype=torch.float32).unsqueeze(-1)
        hour_of_year = torch.tensor(hour_of_year, dtype=torch.float32).unsqueeze(-1)
        days_in_month = torch.tensor(days_in_month, dtype=torch.float32).unsqueeze(-1)
        days_in_year = torch.tensor(days_in_year, dtype=torch.float32).unsqueeze(-1)

        return {
            'covariates': covariates, 
            'avg_target': avg_target,
            'target': target, 
            'mask': mask, 
            'hour_of_day': hour_of_day,
            'hour_of_week': hour_of_week,
            'hour_of_month': hour_of_month,
            'hour_of_year': hour_of_year,
            'days_in_month': days_in_month,
            'days_in_year': days_in_year,
            'unix_date': unix_date,
            'file': file,
            'key': key,
        }
