import torch
import argparse
from src.models.modulated_siren import ModulatedSirenModel
from src.datamodules.chairs_datamodule import ChairsDatamodule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def main(args):

    seed_everything(1994)

    # Initialize the wandb logger
    wandb_logger_name = args.name
    wandb_logger = WandbLogger(name=wandb_logger_name, project="Functa")

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{wandb_logger_name}', save_last=True, every_n_epochs=100)
    early_stopping = EarlyStopping(
      monitor='train_loss',  # Metric to monitor
      patience=100,           # Number of epochs with no improvement
      verbose=True,          # Print messages when stopping
      mode='min'             # Minimize the monitored metric
    )

    train_loader = ChairsDatamodule(path= "/home/arkadi.piven/Code/functa/rendered/chair", batch_size=args.batch_size)

    # Initialize the model
    model = ModulatedSirenModel(in_features=2, hidden_features=256, hidden_layers=9, out_features=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.)

    # Initialize a trainer
    trainer = Trainer(logger=wandb_logger,
                       callbacks=[checkpoint_callback, early_stopping],
                         max_epochs=args.max_epochs,
                           gpus=1 if torch.cuda.is_available() else 0,
                             log_every_n_steps=1)

    trainer.fit(model, train_loader)

def arg_parser():
    parser = argparse.ArgumentParser(description='Train Modulated Siren Model')

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1000)


    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    main(args)