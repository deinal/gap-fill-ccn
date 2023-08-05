import os
import json
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from gapt.constants import feature_list
from gapt.data import GapFillingDataset
from gapt.model import GapT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--data_splits', default=[0.8, 0.1, 0.1], nargs=3, type=float)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()

    # Create the dataset
    dataset = GapFillingDataset(args.data_dir, feature_list)

    # Split the dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, args.data_splits, generator=generator)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the model
    n_input = len(feature_list)
    n_output = 1

    model = GapT(
        n_input,
        args.n_head,
        args.n_hidden,
        args.n_layers,
        n_output,
        args.learning_rate
    )

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
        'training_directories': train_dataset.indices,
        'validation_directories': val_dataset.indices,
        'test_directories': test_dataset.indices,
        'loss_values': results
    }

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
