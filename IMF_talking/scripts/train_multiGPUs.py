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
from IMF_talking.trainer.IMFTrainer import IMFTrainer
from IMF_talking.datasets.video_dataset import VideoDataset, VideoCollate
from IMF_talking.utils.training_utils import add_gradient_hooks
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        # init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

def cleanup_ddp():
    dist.destroy_process_group()

def load_config(config_path):
    return OmegaConf.load(config_path)

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_default.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--recompute_dataset', action='store_true')
    return parser.parse_args()

def main(rank, world_size):
    # rank = int(os.getenv("LOCAL_RANK", 0))  # 로컬에서의 GPU ID
    # rank = dist.get_rank()  # 글로벌 프로세스 ID
    # world_size = dist.get_world_size()  # 전체 프로세스 수

    print(f"Rank: {rank}, World Size: {world_size}")
    # torch.cuda.set_device(rank)

    setup_ddp(rank, world_size)

    args = get_arg_parser()
    config = load_config(args.config)

    torch.cuda.empty_cache()
    print("Rank: ", rank)
    if rank == 0 and not args.debug: # if not in debug mode, use wandb
        wandb.init(project='IMF', config=OmegaConf.to_container(config, resolve=True))
        config.debug = False
    else:
        config.debug = True

    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        cpu=config.accelerator.cpu
    )

    print("using float32")
    torch.set_default_dtype(torch.float32)

    model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature
    ).to(rank)
    add_gradient_hooks(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)


    # discriminator = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
    # discriminator = LiaDiscriminator(size=256,channel_multiplier=1)
    discriminator = IMFPatchDiscriminator().to(rank)
    add_gradient_hooks(discriminator)
    discriminator = DDP(discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    transform = transforms.Compose([
    ])

    print("Model initialized successfully")

    dataset = VideoDataset(
        video_folder_list_txt_file=config.dataset.video_folder_list_txt_file,
        recompute=args.recompute_dataset,
        n_frames=(config.training.batch_size * 8),
        target_fps=None,
        target_resolution=config.dataset.target_resolution,
        rank=rank,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset,
        # batch_size=config.training.batch_size,
        batch_size=1, # we will use batch size inside the IMFTrainer.
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=VideoCollate(
            transform=transform,
            _min=0,
            _max=1), # we normalize the video frames to [0, 1] inside the collate function
        sampler=sampler,
    )

    trainer = IMFTrainer(config, model, discriminator, dataloader, accelerator, rank=rank)
    # Check if a checkpoint path is provided in the config
    if config.training.load_checkpoint:
        checkpoint_path = config.training.checkpoint_path
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 0
    trainer.train(start_epoch)

if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

    cleanup_ddp()

    # torchrun --nproc_per_node=N train_multiGPUs.py --config configs/config_default.yaml
    #  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/train_multiGPUs.py --config configs/config_default.yaml