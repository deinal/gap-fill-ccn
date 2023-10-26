import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from modules.utils import periodic_positional_encoding
from modules.layers import TCN, MLP, Time2Vec
from momo import Momo


class GapT(pl.LightningModule):
    def __init__(self, encoder_type, mode, time_encoding,
                 n_layers, n_head, kernel_size,
                 d_input, d_model, d_hidden, d_output, 
                 enc_dropout, ff_dropout,
                 learning_rate, optimizer, 
                 log_scaled=True):
        super().__init__()

        self.encoder_type = encoder_type
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.mode = mode
        self.time_encoding = time_encoding
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
        
        if time_encoding:
            assert d_model % 4 == 0, 'd_embedding should be divisible by number of time features'

        if time_encoding == 'time2vec':
            self.time2vec_day = Time2Vec(1, self.d_embedding // 4)
            self.time2vec_week = Time2Vec(1, self.d_embedding // 4)
            self.time2vec_month = Time2Vec(1, self.d_embedding // 4)
            self.time2vec_year = Time2Vec(1, self.d_embedding // 4)
        
        if encoder_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_hidden, 
                dropout=enc_dropout,
                activation='gelu',
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                transformer_layer,
                num_layers=n_layers
            )
            ff_input = d_model
        elif encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=d_model, 
                hidden_size=d_hidden,
                num_layers=n_layers,
                dropout=enc_dropout,
                batch_first=True,
                bidirectional=True,
            )
            ff_input = 2 * d_hidden
        elif encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_hidden,
                num_layers=n_layers,
                dropout=enc_dropout,
                batch_first=True,
                bidirectional=True,
            )
            ff_input = 2 * d_hidden
        elif encoder_type == 'tcn':
            self.encoder = TCN(
                input_size=d_model,
                hidden_size=d_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size,
                dropout=enc_dropout,
            )
            ff_input = d_hidden
        elif encoder_type == 'mlp':
            self.encoder = MLP(
                input_size=d_model,
                hidden_size=d_hidden,
                num_layers=n_layers,
                dropout=enc_dropout,
            )
            ff_input = d_hidden
        else:
            raise ValueError('Invalid encoder')
        
        self.feed_forward = nn.Sequential(
            nn.Linear(ff_input, 128),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(ff_dropout),
        )

        self.head = nn.Linear(32, d_output)

    def forward(self, batch):
        if self.time_encoding == 'time2vec':
            day_enc = self.time2vec_day(batch['hour_of_day'])
            week_enc = self.time2vec_week(batch['hour_of_week'])
            month_enc = self.time2vec_month(batch['hour_of_month'])
            year_enc = self.time2vec_year(batch['hour_of_year'])
            positional_encoding = torch.cat([day_enc, week_enc, month_enc, year_enc], dim=-1)
        elif self.time_encoding == 'periodic':
            day_enc = periodic_positional_encoding(batch['hour_of_day'], self.d_embedding // 4, period=24)
            week_enc = periodic_positional_encoding(batch['hour_of_week'], self.d_embedding // 4, period=7*24)
            month_enc = periodic_positional_encoding(batch['hour_of_month'], self.d_embedding // 4, period=batch['days_in_month']*24)
            year_enc = periodic_positional_encoding(batch['hour_of_year'], self.d_embedding // 4, period=batch['days_in_year']*24)
            positional_encoding = torch.cat([day_enc, week_enc, month_enc, year_enc], dim=-1)

        if self.mode == 'default':
            # Embed covariates and target separately
            embedded_cov = self.embedding_cov(batch['covariates'])
            embedded_tgt = self.embedding_tgt(batch['avg_target'])
            
            # Add encoded time
            if self.time_encoding:
                embedded_cov = self.embedding_cov(embedded_cov) + positional_encoding
                embedded_tgt = self.embedding_tgt(embedded_tgt) + positional_encoding
            
            # Concatenate the embeddings
            embedding = torch.cat([embedded_cov, embedded_tgt], dim=-1)
        elif self.mode == 'naive':
            # Concatenate covariates and target
            inputs = torch.cat([batch['covariates'], batch['avg_target']], dim=-1)

            # Embed inputs 
            embedding = self.embedding(inputs)

            # Add encoded time
            if self.time_encoding:
                embedding = embedding + positional_encoding
        else:
            raise ValueError('Invalid mode')
        
        # Apply encoder
        if self.encoder_type in ['lstm', 'gru']:
            output, _ = self.encoder(embedding)
        else:
            output = self.encoder(embedding)

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