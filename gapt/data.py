import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset
from gapt.constants import gases, met, aerosols


class GapFillingDataset(Dataset):
    def __init__(self, data_paths, feature_list):
        self.feature_list = feature_list
        self.data_paths = data_paths
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file, key = self.data_paths[idx]
        
        # Load observation and covariate data from the HDF5 file
        with h5py.File(file, 'r') as h5file:
            observations = pd.DataFrame(h5file[key]['observations'])
            covariates = pd.DataFrame(h5file[key]['covariates'])
            unix_date = h5file[key]['observations'][:,0]

        # Assign column names
        observations.columns = ['date', 'avg_target', 'target', 'mask', 'padded']
        covariates.columns = ['date', 'padded'] + self.feature_list

        # Convert Unix timestamps to pandas datetime objects
        observations['date'] = pd.to_datetime(observations['date'], unit='s')
        covariates['date'] = pd.to_datetime(covariates['date'], unit='s')

        # Extract the features and the target
        cov_features = covariates[self.feature_list].values
        avg_target = observations['avg_target'].values
        target = observations['target'].values
        mask = observations['mask'].values
        minutes_observations = observations['date'].dt.hour * 60 + observations['date'].dt.minute.values
        minutes_covariates = covariates['date'].dt.hour * 60 + covariates['date'].dt.minute.values
        padded_observations = observations['padded'].values
        padded_covariates = covariates['padded'].values

        # Convert to tensors
        cov_features = torch.tensor(cov_features, dtype=torch.float32)
        avg_target = torch.tensor(avg_target, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(-1)
        minutes_observations = torch.tensor(minutes_observations, dtype=torch.float32).unsqueeze(-1)
        minutes_covariates = torch.tensor(minutes_covariates, dtype=torch.float32).unsqueeze(-1)
        padded_observations = torch.tensor(padded_observations, dtype=torch.bool).unsqueeze(-1)
        padded_covariates = torch.tensor(padded_covariates, dtype=torch.bool).unsqueeze(-1)

        return {
            'covariates': cov_features, 
            'avg_target': avg_target,
            'target': target, 
            'mask': mask, 
            'minutes_observations': minutes_observations, 
            'minutes_covariates': minutes_covariates, 
            'padded_observations': padded_observations, 
            'padded_covariates': padded_covariates,
            'unix_date': unix_date,
            'file': file,
            'key': key,
        }
