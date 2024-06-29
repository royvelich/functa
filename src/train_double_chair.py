import torch
import argparse
from src.models.double_modulated_siren import DoubleModulatedSirenModel
from src.datamodules.chairs_datamodule import ChairsDatamodule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy


def main(args):

    seed_everything(1994)

    # Initialize the wandb logger
    wandb_logger_name = args.name
    wandb_logger = WandbLogger(name=wandb_logger_name, project="Functa")

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{wandb_logger_name}', save_last=True, every_n_epochs=100)

    train_loader = ChairsDatamodule(path= "/home/arkadi.piven/Code/functa/rendered/chair", dim=args.dim, batch_size=args.batch_size, double=True)

    strategy = DDPStrategy(find_unused_parameters=False)

    # Initialize the model
    model = DoubleModulatedSirenModel(in_features=2,
                                hidden_features=args.hidden_features,
                                hidden_layers=args.hidden_layers,
                                modulation_size=args.modulation_size,
                                out_features=3,
                                outermost_linear=True,
                                first_omega_0=30,
                                hidden_omega_0=30.,
                                lr=args.lr,
                                epochs=args.max_epochs
                                )

    # Initialize a trainer

    if args.ddp:
      trainer = Trainer(logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        max_epochs=args.max_epochs,
                        accelerator='gpu',
                        gpus=-1 if torch.cuda.is_available() else 0,
                        log_every_n_steps=1,
                        strategy=strategy
                        )

    else:
      trainer = Trainer(logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        max_epochs=args.max_epochs,
                        gpus=1 if torch.cuda.is_available() else 0,
                        log_every_n_steps=1
                        )

    if args.checkpoint:
        model = DoubleModulatedSirenModel.load_from_checkpoint(args.checkpoint, in_features=2, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers, modulation_size=args.modulation_size, out_features=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.)
    trainer.fit(model, train_loader)

def arg_parser():
    parser = argparse.ArgumentParser(description='Train Modulated Siren Model')

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--hidden_layers', type=int, default=9)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--modulation_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=str, default=None)


    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    main(args)