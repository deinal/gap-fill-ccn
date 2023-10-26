import os
import json
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modules.constants import feature_list
from modules.data import GapFillingDataset
from modules.gapt import GapT
from modules.baseline import Baseline
from modules.rnn import RNN
from modules.mlp import MLP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gapt', choices=['gapt', 'baseline'])
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'naive'])
    parser.add_argument('--encoder_type', type=str, default='transformer',  choices=['transformer', 'lstm', 'gru', 'tcn', 'mlp'])
    parser.add_argument('--time_encoding', type=str, default='time2vec', choices=['time2vec', 'periodic', 'none'])
    parser.add_argument('--d_output', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_embedding', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--d_hidden', type=int, default=128)
    parser.add_argument('--enc_dropout', type=float, default=0.0)
    parser.add_argument('--ff_dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='momo', choices=['momo', 'adam'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # Create the datasets
    with open(os.path.join(args.data_dir, 'paths.json'), 'r') as f:
        data_paths = json.load(f)
    
    train_dataset = GapFillingDataset(data_paths['train'], feature_list)
    val_dataset = GapFillingDataset(data_paths['val'], feature_list)
    test_dataset = GapFillingDataset(data_paths['test'], feature_list)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the model
    d_input = len(feature_list)

    if args.model == 'gapt':
        model = GapT(
            encoder_type=args.encoder_type,
            mode=args.mode,
            time_encoding=args.time_encoding,
            d_input=d_input,
            d_model=args.d_model,
            n_head=args.n_head,
            kernel_size=args.kernel_size,
            d_hidden=args.d_hidden,
            n_layers=args.n_layers,
            d_output=args.d_output, 
            enc_dropout=args.enc_dropout,
            ff_dropout=args.ff_dropout,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            
        )
    elif args.model == 'baseline':
        model = Baseline(
            d_input=d_input,
            d_embedding=args.d_embedding,
            d_model=args.d_model,
            d_output=args.d_output,
            learning_rate=args.learning_rate,
            dropout_rate=args.ff_dropout,
            optimizer=args.optimizer,
        )
    else:
        raise ValueError(f'Invalid model type: {args.model}')

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=args.devices,
        max_epochs=args.epochs, 
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger('logs/')
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test the model
    results = trainer.test(dataloaders=[train_dataloader, val_dataloader, test_dataloader])

    # Save the model
    trainer.save_checkpoint(os.path.join(args.output_dir, 'model.ckpt'))

    # Save metadata
    metadata = {
        'args': vars(args),
        'feature_list': feature_list,
        'loss_values': results,
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
