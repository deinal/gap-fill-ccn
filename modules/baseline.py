import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from modules.utils import baseline_positional_encoding
from modules.layers import ScaledDotProductAttention
from momo import Momo


class Baseline(pl.LightningModule):
    def __init__(self, d_input, d_embedding, d_model, d_output, 
                 learning_rate, dropout_rate, optimizer, scaler_params=None):
        super().__init__()

        self.d_embedding = d_embedding
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scaler_params = scaler_params

        self.embedding_cov = nn.Linear(d_input, d_embedding)
        self.embedding_cat = nn.Linear(d_input + d_output, d_embedding)
        self.embedding_tgt = nn.Linear(d_output, d_embedding)
        
        self.attention_cov = ScaledDotProductAttention(d_embedding, d_model)
        self.attention_cat = ScaledDotProductAttention(d_embedding, d_model)
        self.attention_tgt = ScaledDotProductAttention(d_embedding, d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(3 * d_model, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.head = nn.Linear(32, d_output)
        
    def forward(self, batch):
        positional_encoding = baseline_positional_encoding(batch['minutes'], self.d_embedding)

        # Embedding inputs
        embedded_cov = self.embedding_cov(batch['covariates']) + positional_encoding 
        embedded_cat = self.embedding_cat(torch.cat([batch['covariates'], batch['avg_target']], dim=-1)) + positional_encoding
        embedded_tgt = self.embedding_tgt(batch['avg_target']) + positional_encoding
        
        # Scaled Dot Product Attention
        output_cov = self.attention_cov(query=embedded_cov, key=embedded_cov, value=embedded_cov)
        output_cat = self.attention_cat(query=embedded_cat, key=embedded_cat, value=embedded_cat)
        output_tgt = self.attention_tgt(query=embedded_tgt, key=embedded_tgt, value=embedded_tgt)

        # Concatenate the outputs
        concatenated_output = torch.cat([output_cov, output_cat, output_tgt], dim=-1)

        # Feed forward
        output = self.feed_forward(concatenated_output)
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

    def inverse_transform(self, tensor, feature):
        mean = self.scaler_params[feature]['mean']
        std = self.scaler_params[feature]['std']
        return tensor * std + mean

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

        # Invert standard scaling
        prediction_original = self.inverse_transform(prediction, 'target')
        target_original = self.inverse_transform(target, 'target')

        loss = F.mse_loss(prediction_original, target_original)
        self.log('test_loss_original', loss)
        rmse = torch.sqrt(loss)
        self.log('test_rmse_original', rmse)
        mae = F.l1_loss(prediction_original, target_original)
        self.log('test_mae_original', mae)
        mbe = torch.mean(prediction_original - target_original)
        self.log('test_mbe_original', mbe)
    
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