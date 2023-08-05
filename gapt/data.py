import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class GapFillingDataset(Dataset):
    def __init__(self, directory, feature_list):
        self.directories = [f.path for f in os.scandir(directory) if f.is_dir()]
        self.feature_list = feature_list
    
    def __len__(self):
        return len(self.directories)
    
    def __getitem__(self, idx):
        directory = self.directories[idx]
        
        # Load observation and covariate data
        observations = pd.read_csv(os.path.join(directory, 'observations.csv'), parse_dates=['date'])
        covariates = pd.read_csv(os.path.join(directory, 'covariates.csv'), parse_dates=['date'])

        # Extract the features and the target
        features_covariates = covariates[self.feature_list].values
        interpolated_target = observations['interpolated_target'].values
        target = observations['target'].values
        mask = observations['mask'].values
        minutes_observations = observations['date'].dt.hour * 60 + observations['date'].dt.minute.values
        minutes_covariates = covariates['date'].dt.hour * 60 + covariates['date'].dt.minute.values
        padded_observations = observations['padded'].values
        padded_covariates = covariates['padded'].values

        # Convert to tensors
        features_covariates = torch.tensor(features_covariates, dtype=torch.float32)
        interpolated_target = torch.tensor(interpolated_target, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(-1)
        minutes_observations = torch.tensor(minutes_observations, dtype=torch.float32).unsqueeze(-1)
        minutes_covariates = torch.tensor(minutes_covariates, dtype=torch.float32).unsqueeze(-1)
        padded_observations = torch.tensor(padded_observations, dtype=torch.bool)
        padded_covariates = torch.tensor(padded_covariates, dtype=torch.bool)

        return {'covariates': features_covariates, 
                'interpolated_target': interpolated_target,
                'target': target, 
                'mask': mask, 
                'minutes_observations': minutes_observations, 
                'minutes_covariates': minutes_covariates, 
                'padded_observations': padded_observations, 
                'padded_covariates': padded_covariates}
