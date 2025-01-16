import matplotlib.pyplot as plt
import io
import wandb
from PIL import Image
import numpy as np
import torch
import os
from torchvision.utils import save_image

def check_gradient_issues(grads, layers):
    issues = []
    mean_grad = np.mean(grads)
    std_grad = np.std(grads)
    
    for layer, grad in zip(layers, grads):
        if grad > mean_grad + 3 * std_grad:
            issues.append(f"🔥 Potential exploding gradient in {layer}: {grad:.2e}")
        elif grad < mean_grad - 3 * std_grad:
            issues.append(f"🥶 Potential vanishing gradient in {layer}: {grad:.2e}")
    
    if issues:
        return "<br>".join(issues)
    else:
        return "✅ No significant gradient issues detected"


# Global variable to store the current table structure
current_table_columns = None
def log_grad_flow(named_parameters, global_step):
    global current_table_columns

    grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            grads.append(p.grad.abs().mean().item())
    
    if not grads:
        print("No valid gradients found for logging.")
        return
    
    # Normalize gradients
    max_grad = max(grads)
    if max_grad == 0:
        print("👿👿👿 Warning: All gradients are zero. 👿👿👿")
        normalized_grads = grads  # Use unnormalized grads if max is zero
        raise ValueError(f"👿👿👿 Warning: All gradients are zero. 👿👿👿")
    else:
        normalized_grads = [g / max_grad for g in grads]

    # Create the matplotlib figure
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(grads)), normalized_grads, alpha=0.5)
    plt.xticks(range(len(grads)), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title(f"Gradient Flow (Step {global_step})")
    if max_grad == 0:
        plt.title(f"Gradient Flow (Step {global_step}) - All Gradients Zero")
    plt.tight_layout()

    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Create a wandb.Image from the buffer
    img = wandb.Image(Image.open(buf))
    
    plt.close()

    # Calculate statistics
    stats = {
        "max_gradient": max_grad,
        "min_gradient": min(grads),
        "mean_gradient": np.mean(grads),
        "median_gradient": np.median(grads),
        "gradient_variance": np.var(grads),
    }

    # Check for gradient issues
    issues = check_gradient_issues(grads, layers)

    # Log everything
    log_dict = {
        "gradient_flow_plot": img,
        **stats,
        "gradient_issues": wandb.Html(issues),
        "step": global_step
    }

    # Log other metrics
    wandb.log(log_dict)


def sample_recon(data, accelerator, output_path, num_samples=1, is_debug=False):
    with torch.no_grad():
        try:
            x_reconstructed,x_current, x_reference = data
            batch_size = x_reconstructed.size(0)
            num_samples = min(num_samples, batch_size)
            
            # Select a subset of images if batch_size > num_samples
            x_reconstructed = x_reconstructed[:num_samples]
            x_reference = x_reference[:num_samples]
            x_current = x_current[:num_samples]

            # Prepare frames for saving (2 rows: clamped reconstructed and original reference)
            frames = torch.cat((x_reconstructed,x_current, x_reference), dim=0)
            
            # Ensure we have a valid output directory
            if output_path:
                output_dir = os.path.dirname(output_path)
                if not output_dir:
                    output_dir = '.'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save frames as a grid (2 rows, num_samples columns)
                save_image(accelerator.gather(frames), output_path, nrow=num_samples, padding=2, normalize=False)
                # accelerator.print(f"Saved sample reconstructions to {output_path}")
            else:
                accelerator.print("Warning: No output path provided. Skipping image save.")
            if not is_debug:
                # Log images to wandb
                wandb_images = []
                for i in range(num_samples):
                    wandb_images.extend([
                        wandb.Image(x_reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reconstructed {i}"),
                        wandb.Image(x_current[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_current {i}"),
                        wandb.Image(x_reference[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reference {i}")
                    ])
                
                wandb.log({"Sample Reconstructions": wandb_images})
            
            elif is_debug:
                # Save images to debug directory
                x_reconstructed = x_reconstructed.cpu()
                x_current = x_current.cpu()
                x_reference = x_reference.cpu()
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                for i in range(num_samples):
                    save_image(x_reconstructed[i], os.path.join(debug_dir, f"x_reconstructed_{i}.png"))
                    save_image(x_current[i], os.path.join(debug_dir, f"x_current_{i}.png"))
                    save_image(x_reference[i], os.path.join(debug_dir, f"x_reference_{i}.png"))
                
                accelerator.print(f"Saved sample reconstructions to {debug_dir}")

            return frames
        except RuntimeError as e:
            print(f"🔥 e:{e}")
        return None