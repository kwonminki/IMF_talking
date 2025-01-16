import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image
from tqdm.auto import tqdm
import wandb
from IMF_talking.utils.wandb_utils import log_grad_flow, sample_recon
from IMF_talking.utils.tensor_utils import imagenet_normalize
from IMF_talking.utils.training_utils import copy_all_files, find_package_path, get_layer_wise_learning_rates
import os
import datetime

def consistent_sub_sample(tensor1, tensor2, sub_sample_size):
    """
    Consistently sub-sample two tensors with the same random offset.
    
    Args:
    tensor1 (torch.Tensor): First input tensor of shape (B, C, H, W)
    tensor2 (torch.Tensor): Second input tensor of shape (B, C, H, W)
    sub_sample_size (tuple): Desired sub-sample size (h, w)
    
    Returns:
    tuple: Sub-sampled versions of tensor1 and tensor2
    """
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    assert tensor1.ndim == 4, "Input tensors should have 4 dimensions (B, C, H, W)"
    
    batch_size, channels, height, width = tensor1.shape
    sub_h, sub_w = sub_sample_size
    
    assert height >= sub_h and width >= sub_w, "Sub-sample size should not exceed the tensor dimensions"
    
    offset_x = torch.randint(0, height - sub_h + 1, (1,)).item()
    offset_y = torch.randint(0, width - sub_w + 1, (1,)).item()
    
    tensor1_sub = tensor1[..., offset_x:offset_x+sub_h, offset_y:offset_y+sub_w]
    tensor2_sub = tensor2[..., offset_x:offset_x+sub_h, offset_y:offset_y+sub_w]
    
    return tensor1_sub, tensor2_sub



class IMFTrainer:
    def __init__(self, config, model, discriminator, train_dataloader, accelerator,
                 optimizers=None, schedulers=None, rank=0):
        self.config = config
        self.model = model
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator
        
        # self.model = self.model.to(self.model.device)
        # self.discriminator = self.discriminator.to(self.model.device)

        self.gan_loss_type = config.loss.type
        # self.perceptual_loss = lpips.LPIPS(net='alex', spatial=True).to(model.device)
        self.perceptual_loss = lpips.LPIPS(net='alex', spatial=True).to(accelerator.device)
        
        def perceptual_loss_fn(x, y):
            # Normalize the input tensors to imagenet mean and std
            x = imagenet_normalize(x)
            y = imagenet_normalize(y)

            return self.perceptual_loss(x, y)

        self.perceptual_loss_fn = perceptual_loss_fn

        self.pixel_loss_fn = nn.L1Loss()
        # self.eye_loss_fn = MediaPipeEyeEnhancementLoss(eye_weight=1.0).to(accelerator.device)


        self.style_mixing_prob = config.training.style_mixing_prob
        self.noise_magnitude = config.training.noise_magnitude
        self.r1_gamma = config.training.r1_gamma

        if optimizers is None:
            self.optimizer_g = Adam(get_layer_wise_learning_rates(model), lr=2e-4, betas=(0.5, 0.999))
            self.optimizer_d = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        else:
            self.optimizer_g = optimizers['optimizer_g']
            self.optimizer_d = optimizers['optimizer_d']

        if schedulers is None:
            # Learning rate schedulers
            total_steps = config.training.num_epochs * len(train_dataloader)
            self.scheduler_g = OneCycleLR(self.optimizer_g, max_lr=2e-4, total_steps=total_steps)
            self.scheduler_d = OneCycleLR(self.optimizer_d, max_lr=2e-4, total_steps=total_steps)
        else:
            self.scheduler_g = schedulers['scheduler_g']
            self.scheduler_d = schedulers['scheduler_d']


        if config.training.use_ema:
            # from stylegan import EMA
            # self.ema = EMA(model, decay=config.training.ema_decay)
            raise NotImplementedError("EMA is not implemented yet")
        else:
            self.ema = None

        self.model, self.discriminator, self.optimizer_g, self.optimizer_d, self.train_dataloader = accelerator.prepare(
            self.model, self.discriminator, self.optimizer_g, self.optimizer_d, self.train_dataloader
        )
        if self.ema:
            self.ema = accelerator.prepare(self.ema)
            self.ema.register()

        self.output_dir = os.path.join("outputs", self.config.project_name)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(self.output_dir, f"{timestamp}")

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.val_dir = os.path.join(self.output_dir, "val_images")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        self.rank = rank

        if self.rank == 0:
            copy_all_files(find_package_path('IMF_talking'), 
                            os.path.join(self.output_dir, 'IMF_talking'))

    def check_exploding_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"🔥 Exploding gradients detected in {name}")
                    return True
        return False

    def train_step(self, x_current, x_reference, global_step):
        if x_current.nelement() == 0:
            print("🔥 Skipping training step due to empty x_current")
            return None, None, None, None, None, None

        self.optimizer_g.zero_grad()

        # Generate reconstructed frame
        x_reconstructed = self.model(x_current, x_reference)

        if self.config.training.use_subsampling:
            sub_sample_size = (128, 128)  # As mentioned in the paper https://github.com/johndpope/MegaPortrait-hack/issues/41
            x_current, x_reconstructed = consistent_sub_sample(x_current, x_reconstructed, sub_sample_size)

        # Discriminator updates
        d_loss_total = 0
        for _ in range(self.config.training.n_critic):
            self.optimizer_d.zero_grad()
            
            # Real samples
            real_outputs = self.discriminator(x_current)
            d_loss_real = sum(torch.mean(F.relu(1 - output)) for output in real_outputs)
            
            # Fake samples
            fake_outputs = self.discriminator(x_reconstructed.detach())
            d_loss_fake = sum(torch.mean(F.relu(1 + output)) for output in fake_outputs)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake

            # R1 regularization
            if self.config.training.use_r1_reg and global_step % self.config.training.r1_interval == 0:
                x_current.requires_grad = True
                real_outputs = self.discriminator(x_current)
                r1_reg = 0
                for real_output in real_outputs:
                    grad_real = torch.autograd.grad(
                        outputs=real_output.sum(), inputs=x_current, create_graph=True
                    )[0]
                    r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                d_loss += self.config.training.r1_gamma * r1_reg

            self.accelerator.backward(d_loss)
            
            if self.check_exploding_gradients(self.discriminator):
                print("🔥 Skipping discriminator update due to exploding gradients")
            else:
                if self.config.training.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config.training.clip_grad_norm)
                self.optimizer_d.step()
            
            d_loss_total += d_loss.item()

        # Average discriminator loss
        d_loss_avg = d_loss_total / self.config.training.n_critic

        # Generator update
        fake_outputs = self.discriminator(x_reconstructed)
        g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

        l_p = self.pixel_loss_fn(x_reconstructed, x_current).mean()
        l_v = self.perceptual_loss_fn(x_reconstructed, x_current).mean()
        # l_eye = self.eye_loss_fn(x_reconstructed, x_current) if self.config.training.use_eye_loss else 0

        g_loss = (
            self.config.training.lambda_pixel * l_p +
            self.config.training.lambda_perceptual * l_v +
            self.config.training.lambda_adv * g_loss_gan
                # self.config.training.lambda_eye * l_eye)
            )

        self.accelerator.backward(g_loss)

        if self.check_exploding_gradients(self.model):
            print("🔥 Exploding gradients detected. Clipping gradients.")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        else:
            if self.config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.clip_grad_norm)
        
        self.optimizer_g.step()

        # Step the schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()

        if self.ema:
            self.ema.update()

        # # Logging - locally for sanity check
        # if self.rank == 0 and global_step % self.config.logging.sample_every == 0:
        #     save_image(x_reconstructed, f'x_reconstructed.png', normalize=True)
        #     save_image(x_current, f'x_current.png', normalize=True)
        #     save_image(x_reference, f'x_reference.png', normalize=True)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f"Parameter {name} did not receive gradient.")

        return d_loss_avg, g_loss.item(), l_p.item(), l_v.item(), g_loss_gan.item(), x_reconstructed

    def make_batch_from_frames(self, frames):
        # frames: (1, T, C, H, W)
        assert frames.ndim == 5, "Input frames should have 5 dimensions (B, T, C, H, W)"
        assert frames.shape[0] == 1, "video number should be 1"
        batch_size = self.config.training.batch_size

        # Shuffle the frames
        frames = frames.squeeze(0)
        indices = torch.randperm(frames.size(0))
        frames = frames[indices]

        # Make sure the number of frames is x2 of the batch size
        if frames.shape[0] != batch_size * 2:
            if frames.shape[0] < batch_size * 2:
                while frames.shape[0] < batch_size * 2:
                    frames = torch.cat([frames, frames], dim=0)
            
            frames = frames[:batch_size * 2]

        # Split the frames into two parts
        x_current = frames[:batch_size]
        x_reference = frames[batch_size:]

        return x_current, x_reference


    def train(self, start_epoch=0):
        global_step = start_epoch * len(self.train_dataloader)

        for epoch in range(self.config.training.num_epochs):

            # self.train_dataloader.sampler.set_epoch(epoch) 

            self.model.train()
            self.discriminator.train()
            if self.rank == 0:
                progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")

            epoch_g_loss = 0
            epoch_d_loss = 0
            num_valid_steps = 0
 
            for batch in self.train_dataloader:

                frames = batch['video'].to(self.accelerator.device)

                x_current, x_reference = self.make_batch_from_frames(frames)

                results = self.train_step(x_current, x_reference, global_step)

                if results[0] is not None:
                    d_loss, g_loss, l_p, l_v,  g_loss_gan, x_reconstructed = results
                    epoch_g_loss += g_loss
                    epoch_d_loss += d_loss
                    num_valid_steps += 1

                else:
                    print("Skipping step due to error in train_step")

        
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss

                if self.rank==0 and self.accelerator.is_main_process and global_step % self.config.logging.log_every == 0 and not self.config.debug:
                    wandb.log({
                        "noise_magnitude": self.noise_magnitude,
                        "g_loss": g_loss,
                        "d_loss": d_loss,
                        "pixel_loss": l_p,
                        "perceptual_loss": l_v,
                        "gan_loss": g_loss_gan,
                        "global_step": global_step,
                        "lr_g": self.optimizer_g.param_groups[0]['lr'],
                        "lr_d": self.optimizer_d.param_groups[0]['lr']
                    })
                    # Log gradient flow for generator and discriminator
                    log_grad_flow(self.model.named_parameters(),global_step)
                    log_grad_flow(self.discriminator.named_parameters(),global_step)

                if self.rank == 0 and global_step % self.config.logging.sample_every == 0:
                    sample_path = f"recon_step_{global_step}.png"
                    sample_path = os.path.join(self.val_dir, sample_path)
                    sample_recon((x_reconstructed, x_current, x_reference), self.accelerator, sample_path, 
                                    num_samples=self.config.logging.sample_size, is_debug=self.config.debug)
                    
                global_step += 1

                    # Checkpoint saving
                if self.rank==0 and global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(epoch)

                # # Calculate average losses for the epoch
                # if num_valid_steps > 0:
                #     avg_g_loss = epoch_g_loss / num_valid_steps
                #     avg_d_loss = epoch_d_loss / num_valid_steps

                if self.rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"G Loss": f"{g_loss:.4f}", "D Loss": f"{d_loss:.4f}"})

            if self.rank == 0:
                progress_bar.close()
            

        # Final model saving
        if self.rank == 0:
            self.save_checkpoint(epoch, is_final=True)


    def save_checkpoint(self, epoch, is_final=False):
        self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized
        
        # DDP 또는 Accelerator로 래핑된 모델의 원본 모델 가져오기
        unwrapped_model = (
            self.accelerator.unwrap_model(self.model)
            if hasattr(self.accelerator, "unwrap_model")
            else (self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model)
        )
        unwrapped_discriminator = (
            self.accelerator.unwrap_model(self.discriminator)
            if hasattr(self.accelerator, "unwrap_model")
            else (self.discriminator.module if isinstance(self.discriminator, torch.nn.parallel.DistributedDataParallel) else self.discriminator)
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'discriminator_state_dict': unwrapped_discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
        }
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        save_path = f"{'final_model' if is_final else f'checkpoint_{epoch}'}.pth"
        save_path = os.path.join(self.checkpoint_dir, save_path)
        self.accelerator.save(checkpoint, save_path)  # Save the checkpoint safely
        print(f"Saved checkpoint for epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)

            # DDP 또는 Accelerator로 래핑된 모델의 원본 모델 가져오기
            unwrapped_model = (
                self.accelerator.unwrap_model(self.model)
                if hasattr(self.accelerator, "unwrap_model")
                else (self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model)
            )
            unwrapped_discriminator = (
                self.accelerator.unwrap_model(self.discriminator)
                if hasattr(self.accelerator, "unwrap_model")
                else (self.discriminator.module if isinstance(self.discriminator, torch.nn.parallel.DistributedDataParallel) else self.discriminator)
            )

            # Checkpoint에서 상태 불러오기
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
            unwrapped_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

            if self.ema and 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {start_epoch - 1}")
            return start_epoch
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}")
            return 0
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0