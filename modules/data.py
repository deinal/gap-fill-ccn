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
        
        # Load observation and covariate data from the HDF5 file
        with h5py.File(file, 'r') as h5file:
            observations = pd.DataFrame(h5file[key]['observations'])
            covariates = pd.DataFrame(h5file[key]['covariates'])
            unix_date = h5file[key]['observations'][:,0]

        # Assign column names
        observations.columns = ['date', 'avg_target', 'target', 'mask']
        covariates.columns = ['date'] + self.feature_list

        # Convert Unix timestamps to pandas datetime objects
        observations['date'] = pd.to_datetime(observations['date'], unit='s')
        covariates['date'] = pd.to_datetime(covariates['date'], unit='s')

        # Extract the features and the target
        cov_features = covariates[self.feature_list].values
        avg_target = observations['avg_target'].values
        target = observations['target'].values
        mask = observations['mask'].values
        minutes = observations['date'].dt.hour * 60 + observations['date'].dt.minute.values

        # Convert to tensors
        cov_features = torch.tensor(cov_features, dtype=torch.float32)
        avg_target = torch.tensor(avg_target, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(-1)
        minutes = torch.tensor(minutes, dtype=torch.float32).unsqueeze(-1)

        return {
            'covariates': cov_features, 
            'avg_target': avg_target,
            'target': target, 
            'mask': mask, 
            'minutes': minutes,
            'unix_date': unix_date,
            'file': file,
            'key': key,
        }
