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
        hours = target_data['date'].dt.hour

        # Convert to tensors
        covariates = torch.tensor(covariates, dtype=torch.float32)
        avg_target = torch.tensor(avg_target, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(-1)
        hours = torch.tensor(hours, dtype=torch.float32).unsqueeze(-1)

        return {
            'covariates': covariates, 
            'avg_target': avg_target,
            'target': target, 
            'mask': mask, 
            'hours': hours,
            'unix_date': unix_date,
            'file': file,
            'key': key,
        }
