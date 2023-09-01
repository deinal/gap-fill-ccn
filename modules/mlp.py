import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from momo import Momo


class MLP(pl.LightningModule):
    def __init__(self, d_input, d_output, learning_rate,
                 dropout_rate, optimizer, log_scaled=True):
        super(MLP, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.log_scaled = log_scaled

        # Define the MLP structure
        self.layers = nn.Sequential(
            nn.Linear(d_input + d_output, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, d_output)
        )

    def forward(self, batch):
        # Concatenate covariates and target
        x = torch.cat([batch['covariates'], batch['avg_target']], dim=-1)
        # Pass through feedforward block
        x = self.layers(x)

        # Masking and output
        output = x * ~batch['mask']
        output += batch['avg_target'] * batch['mask']
        return output

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        inverted_mask = ~batch['mask']
        loss = F.mse_loss(outputs[inverted_mask], batch['target'][inverted_mask])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        inverted_mask = ~batch['mask']
        loss = F.mse_loss(outputs[inverted_mask], batch['target'][inverted_mask])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.forward(batch)
        inverted_mask = ~batch['mask']

        prediction = outputs[inverted_mask]
        target = batch['target'][inverted_mask]
        
        loss = F.mse_loss(prediction, target)
        self.log('test_loss', loss)
        rmse = torch.sqrt(loss)
        self.log('test_rmse', rmse)
        mae = F.l1_loss(prediction, target)
        self.log('test_mae', mae)
        mbe = torch.mean(prediction - target)
        self.log('test_mbe', mbe)

        if self.log_scaled:
            # Revert log scaling
            exp_prediction = torch.exp(prediction)
            exp_target = torch.exp(target)
            
            exp_loss = F.mse_loss(exp_prediction, exp_target)
            self.log('test_loss_original', exp_loss)
            exp_rmse = torch.sqrt(exp_loss)
            self.log('test_rmse_original', exp_rmse)
            exp_mae = F.l1_loss(exp_prediction, exp_target)
            self.log('test_mae_original', exp_mae)
            exp_mbe = torch.mean(exp_prediction - exp_target)
            self.log('test_mbe_original', exp_mbe)
    
    def lr_lambda(self, current_epoch):
        max_epochs = self.trainer.max_epochs
        start_decay_epoch = int(max_epochs * 0.7) # Start decay at 70% of max epochs

        if current_epoch < start_decay_epoch:
            return 1.0
        else:
            # Compute the decay factor in the range [0, 1]
            decay_factor = (current_epoch - start_decay_epoch) / (max_epochs - start_decay_epoch)
            # Exponential decay down to 1% of the initial learning rate
            return (1.0 - 0.99 * decay_factor)
    
    def configure_optimizers(self):
        if self.optimizer == 'momo':
            optimizer = Momo(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')

        # scheduler = LambdaLR(optimizer, self.lr_lambda)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'epoch',
            #     'frequency': 1,
            # },
        }