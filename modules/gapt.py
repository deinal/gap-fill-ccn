import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from modules.utils import cyclic_positional_encoding
from momo import Momo

class GapT(pl.LightningModule):
    def __init__(self, d_input, d_model, n_head, d_feedforward, n_layers, d_output, 
                 learning_rate, dropout_rate, optimizer, mode, log_scaled=True):
        super().__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.mode = mode
        self.log_scaled = log_scaled
        
        if mode == 'default':
            assert d_model % 2 == 0, 'd_model should be even'
            self.d_embedding = d_model // 2
            self.embedding_cov = nn.Linear(d_input, self.d_embedding)
            self.embedding_tgt = nn.Linear(d_output, self.d_embedding)
        elif mode == 'naive':
            self.d_embedding = d_model
            self.embedding = nn.Linear(d_input + d_output, self.d_embedding)
        else:
            raise ValueError('Invalid mode')
        
        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward, 
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer_enc = nn.TransformerEncoder(transformer_enc_layer, num_layers=n_layers)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.head = nn.Linear(32, d_output)

    def forward(self, batch):
        positional_encoding = cyclic_positional_encoding(batch['minutes'], self.d_embedding)

        if self.mode == 'default':
            # Embed covariates and target separately
            embedded_cov = self.embedding_cov(batch['covariates']) + positional_encoding
            embedded_tgt = self.embedding_tgt(batch['avg_target']) + positional_encoding
            # Concatenate the embeddings
            embedding = torch.cat([embedded_cov, embedded_tgt], dim=-1)
        elif self.mode == 'naive':
            # Concatenate covariates and target
            inputs = torch.cat([batch['covariates'], batch['avg_target']], dim=-1)
            # Embed inputs 
            embedding = self.embedding(inputs) + positional_encoding
        else:
            raise ValueError('Invalid mode')
        
        # Apply transformer encoder
        output = self.transformer_enc(embedding)

        # Feed forward
        output = self.feed_forward(output)
        output = self.head(output)
        
        # Masking and output
        output = output * ~batch['mask']
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