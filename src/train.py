import torch
import argparse
from src.models.modulated_siren import ModulatedSirenModel
from src.datamodules.chairs_datamodule import ChairsDatamodule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(args):

    seed_everything(1994)

    # Initialize the wandb logger
    wandb_logger_name = args.name + '_' + args.train_fold
    wandb_logger = WandbLogger(name=wandb_logger_name, project="Functa")

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{wandb_logger_name}', save_last=True, save_top_k=-1, every_n_epochs=100)

    # Initialize the model
    model = ModulatedSirenModel()

    # Initialize a trainer
    trainer = Trainer(logger=wandb_logger,
                       callbacks=[checkpoint_callback],
                         max_epochs=args.max_epochs,
                           gpus=1 if torch.cuda.is_available() else 0,
                             log_every_n_steps=1)

    trainer.fit(model)

def arg_parser():
    parser = argparse.ArgumentParser(description='Train Modulated Siren Model')

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1000)


    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    main(args)