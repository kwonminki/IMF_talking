import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import yaml
import os
import torch.nn.functional as F
from IMF_talking.models.model import IMFModel,IMFPatchDiscriminator,MultiScalePatchDiscriminator
from IMF_talking.utils.training_utils import get_layer_wise_learning_rates
from IMF_talking.trainer.IMFTrainer import IMFTrainer
from IMF_talking.datasets.video_dataset import VideoDataset, VideoCollate
from IMF_talking.utils.training_utils import add_gradient_hooks
from omegaconf import OmegaConf
from argparse import ArgumentParser


def load_config(config_path):
    return OmegaConf.load(config_path)

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_default.yaml')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():

    args = get_arg_parser()
    config = load_config(args.config)

    torch.cuda.empty_cache()
    if not args.debug: # if not in debug mode, use wandb
        wandb.init(project='IMF', config=OmegaConf.to_container(config, resolve=True))
        config.debug = False
    else:
        config.debug = True

    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        cpu=config.accelerator.cpu
    )

    model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature
    )
    add_gradient_hooks(model)

    # discriminator = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
    # discriminator = LiaDiscriminator(size=256,channel_multiplier=1)
    discriminator = IMFPatchDiscriminator()
    add_gradient_hooks(discriminator)

    transform = transforms.Compose([
    ])

    print("Model initialized successfully")

    dataset = VideoDataset(
        video_folder_list_txt_file=config.dataset.video_folder_list_txt_file,
        recompute=False,
        n_frames=(config.training.batch_size * 8),
        target_fps=None,
        target_resolution=(256, 256),
    )

    dataloader = DataLoader(
        dataset,
        # batch_size=config.training.batch_size,
        batch_size=1, # we will use batch size inside the IMFTrainer.
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        collate_fn=VideoCollate(
            transform=transform,
            _min=0,
            _max=1)
    )

    optimizer = {
        "optimizer_g": optim.Adam(get_layer_wise_learning_rates(model), lr=2e-4, betas=(0.5, 0.999)),
        "optimizer_d": optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    }

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    trainer = IMFTrainer(config, model, discriminator, dataloader, accelerator, optimizers=optimizer)

    print("using float32")
    torch.set_default_dtype(torch.float32)

    trainer = IMFTrainer(config, model, discriminator, dataloader, accelerator)
    # Check if a checkpoint path is provided in the config
    if config.training.load_checkpoint:
        checkpoint_path = config.training.checkpoint_path
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 0
    trainer.train(start_epoch)

if __name__ == "__main__":
    main()
