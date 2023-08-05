import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from gapt.utils import cyclic_positional_encoding


class GapT(pl.LightningModule):
    def __init__(self, n_input, n_head, n_hidden, n_layers, n_output, learning_rate=1e-4):
        super().__init__()
        # Define embeddings
        self.feature_embedding = nn.Linear(n_input, n_hidden)
        self.target_embedding = nn.Linear(1, n_hidden)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=n_hidden, 
            nhead=n_head, 
            num_encoder_layers=n_layers, 
            num_decoder_layers=n_layers,
            dim_feedforward=n_hidden,
            activation='gelu',
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU()
        )
        
        self.head = nn.Linear(32, n_output)

        self.learning_rate = learning_rate
        
    def forward(self, batch):
        # Positional encodings
        feature_positional_encoding = cyclic_positional_encoding(
            batch['minutes_covariates'], self.feature_embedding.out_features)
        target_positional_encoding = cyclic_positional_encoding(
            batch['minutes_observations'], self.target_embedding.out_features)
        
        # Embedding inputs
        embedded_features = self.feature_embedding(batch['covariates']) + feature_positional_encoding # B, S_cov, F
        embedded_target = self.target_embedding(batch['interpolated_target']) + target_positional_encoding # B, S_seq, F
        
        # Transformer
        output = self.transformer(embedded_features, embedded_target,
            src_key_padding_mask=batch['padded_covariates'], # B, S_cov
            tgt_key_padding_mask=batch['padded_observations'])# | ~batch['mask'])) # B, S_seq
        
        output = self.feed_forward(output)
        output = self.head(output) # B, S_seq, 1

        # Masking and output
        output = output * ~batch['mask']
        output += batch['interpolated_target'] * batch['mask']
        return output

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = F.mse_loss(outputs, batch['target'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = F.mse_loss(outputs, batch['target'])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.forward(batch)
        loss = F.mse_loss(outputs, batch['target'])
        self.log('test_loss', loss)
    
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, self.lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }