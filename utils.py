import re
import torch
import shutil
import resource
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torchvision.utils import make_grid




def check_model_output(model, sample_input, device, printmsg=False):
    # check if model output shape is matching the frame shape
    model.eval()
    with torch.no_grad():
        frames, flows, target_frame = sample_input
        frames, flows, target_frame = frames.to(device), flows.to(device), target_frame.to(device)
        try:
            print(f"Frames shape: {frames.shape}") if printmsg else 0
            print(f"Flows shape: {flows.shape}") if printmsg else 0
            print(f"Target frame shape: {target_frame.shape}") if printmsg else 0
            output, mu, logvar = model(frames, flows)
            print(f"Output shape: {output.shape}") if printmsg else 0
            print(f"Mu shape: {mu.shape}") if printmsg else 0
            print(f"Logvar shape: {logvar.shape}") if printmsg else 0
            # print(f"Seg_mask shape: {seg_mask.shape}") if printmsg else 0
            assert output.shape == target_frame.shape, "Model output shape doesn't match target frame shape"
        except Exception as e:
            print(f"Error in model forward pass: {str(e)}") if printmsg else 0
            print(f"Model architecture:") if printmsg else 0
            print(model) if printmsg else 0
            raise


def visualize_frames(frames, masks=None, title="", save_path=None, nrow=4):
    # Convert frames to CPU if they're on GPU
    frames = frames.cpu()
    if masks is not None:
        masks = masks.cpu()
    
    # Get the dimensions of a single frame
    _, h, w = frames[0].shape
    aspect_ratio = w / h
    
    # Calculate the size of the figure
    fig_width = 24  # You can adjust this value to change the overall width of the figure
    fig_height = fig_width / (nrow * aspect_ratio)
    
    # Create the grid for frames
    frame_grid = make_grid(frames, nrow=nrow, normalize=True, padding=2).permute(1, 2, 0).numpy()
    
    if masks is not None:
        # Create the grid for masks
        mask_grid = make_grid(masks, nrow=nrow, normalize=False, padding=2).permute(1, 2, 0).numpy()
        assert torch.all(masks>=0), masks

        # Create a figure for the overlay
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Display the frame grid
        ax.imshow(frame_grid)
        
        # Determine the range of the mask values
        vmin, vmax = 0, +1
        norm = Normalize(vmin=vmin, vmax=vmax)        
        # Overlay the mask grid with a colormap and alpha
        mask_overlay = ax.imshow(mask_grid[:,:,0], cmap='gray', alpha=0.4, norm=norm)

        # Add a colorbar for the mask overlay
        # cbar = plt.colorbar(mask_overlay, ax=ax, label='Segmentation', fraction=0.046, pad=0.04)
        # cbar.set_ticks([vmin, vmax])
        # cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    
        ax.set_title(f"{title}")
    else:
        # If there are no masks, create a single plot of frames
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(frame_grid)
        ax.set_title(title)
    
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_loss(total_losses, recon_losses, kl_divs, motion_losses, save_path=None):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.plot(total_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 4, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 4, 3)
    plt.plot(kl_divs)
    plt.title('KL Divergence')
    plt.xlabel('Epoch')

    plt.subplot(1, 4, 4)
    plt.plot(motion_losses)
    plt.title('Motion Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_checkpoint(model, optimizer, epoch, losses, best_loss, save_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'best_loss': best_loss
    }
    filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['losses'], checkpoint['best_loss']


def get_epoch_number(filename):
    match = re.search(r'checkpoint_epoch_(\d+).pt', filename)
    if match:
        return int(match.group(1))
    else:
        return -1
    
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kl_divergence_clamp(mu, logvar):
    mu = torch.clamp(mu, -10, 10) # Clamp values to prevent extreme outputs
    logvar = torch.clamp(logvar, -10, 10)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()



def increase_file_limit(new_limit=4096):
    # increase max IO file limit for system
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        print(f"Successfully increased file limit to {new_limit}")
    except ValueError as e:
        print(f"Failed to increase file limit: {e}")
        print(f"Current limits - Soft: {soft}, Hard: {hard}")

def check_disk_space(path):
    # how much space is available on disk
    total, used, free = shutil.disk_usage(path)
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free // (2**30)} GB")




def accumulated_motion_loss(recon_frame, target_frame, frames):
    '''
    compare predicted motion for the last frame with
    accumulated actual motion over consecutive frames.
    '''
    motion_loss = 0
    batch_size, sequence_length, channels, height, width = frames.shape
    
    # Calculate actual accumulated motion
    actual_motion = torch.zeros_like(target_frame)
    for i in range(1, sequence_length):
        frame_motion = frames[:, i] - frames[:, i-1]
        actual_motion += frame_motion
    
    # Predicted motion (from last input frame to target frame)
    pred_motion = recon_frame - frames[:, -1]
    
    # Compare predicted motion to accumulated actual motion
    motion_loss = torch.nn.functional.mse_loss(pred_motion, actual_motion, reduction='mean')
    
    return motion_loss


def segmentation_aware_loss(recon_frame, orig_frame, target_frame, segmentation_mask):
    # Compute reconstruction loss
    recon_loss = F.mse_loss(recon_frame, target_frame, reduction='none')
    
    # Compute motion mask (difference between frames)
    motion_threshold = 0.01
    motion_mask = torch.abs(target_frame - orig_frame).mean(dim=1, keepdim=True)
    motion_mask = (motion_mask > motion_threshold).float() # thresholding to reduce noise sensitivity
    
    # Tile the motion mask to match the number of channels in the frame
    motion_mask = motion_mask.repeat(1, 3, 1, 1)
    segmentation_mask = segmentation_mask.repeat(1, 3, 1, 1)

    # Combine segmentation and motion masks
    combined_mask = segmentation_mask + 1.5*motion_mask
    
    # Apply masks to reconstruction loss
    weighted_recon_loss = (recon_loss*combined_mask).sum() + (recon_loss*(1-combined_mask)).sum() * 0.1
    
    return weighted_recon_loss


def emphasized_motion_loss(recon_frame, orig_frame, target_frame, emphasis=1):
    # Compute reconstruction loss per pixel
    recon_loss = F.mse_loss(recon_frame, target_frame, reduction='none')
    
    # Compute motion mask
    motion_threshold = 0.01
    motion_mask = torch.abs(target_frame - orig_frame).mean(dim=1, keepdim=True)
    motion_mask = (motion_mask > motion_threshold).float()
    
    # Tile the motion mask to match the number of channels
    motion_mask = motion_mask.repeat(1, 3, 1, 1)
    
    # Apply motion mask to loss
    weighted_motion_loss = (recon_loss * emphasis * motion_mask).sum() / recon_loss.numel()
    
    return weighted_motion_loss


def weighted_laplacian_pyramid_loss(pred_frame, target_frame, levels=5, level_weights=[16, 8, 4, 2, 1]):
    '''
    Laplacian Pyramid Loss computes differences at multiple scales, 
    which can capture both fine and coarse motion details.
    Fewer levels (earlier in the pyramid) represent finer details and local structures.
    More levels (later in the pyramid) represent coarser structures and global information.
    '''
    def build_laplacian_pyramid(img, levels):
        pyramid = [img]
        for _ in range(levels - 1):
            img = F.avg_pool2d(img, 2)
            pyramid.append(img)
        return pyramid
    
    pred_pyramid = build_laplacian_pyramid(pred_frame, levels)
    target_pyramid = build_laplacian_pyramid(target_frame, levels)
    
    # If no weights are provided, use default weights that emphasize lower levels
    if level_weights is None:
        level_weights = [2**i for i in range(levels)]
    
    # Normalize weights so that they sum to 1
    level_weights = torch.tensor(level_weights, device=pred_frame.device)
    level_weights = level_weights / level_weights.sum()

    loss = 0
    for i, (pred_level, target_level) in enumerate(zip(pred_pyramid, target_pyramid)):
        level_loss = F.l1_loss(pred_level, target_level)
        loss += level_weights[i] * level_loss

    return loss



class BalancedLoss:
    def __init__(self, recon_weight=1.0, beta=1.0, motion_weight=1.0, scale=1, use_log=False, adaptive=False):
        self.recon_weight = recon_weight
        self.beta = beta
        self.motion_weight = motion_weight
        self.scale = scale # scale the total loss for numerical stability
        self.use_log = use_log
        self.adaptive = adaptive
        self.running_recon = None # running avg
        self.running_kl = None
        self.running_motion = None
        self.alpha = 0.99  # smoothness of running average

    def __call__(self, recon_loss, kl_div, motion_loss):
        if torch.isnan(recon_loss) or torch.isnan(kl_div) or torch.isnan(motion_loss):
            print(f"NaN detected in raw losses: Recon: {recon_loss.item()}, KL: {kl_div.item()}, Motion: {motion_loss.item()}")
            return torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))
        if self.running_recon is None:
            self.running_recon = recon_loss.item()
            self.running_kl = kl_div.item()
            self.running_motion = motion_loss.item()
        else:
            self.running_recon = self.alpha * self.running_recon + (1 - self.alpha) * recon_loss.item()
            self.running_kl = self.alpha * self.running_kl + (1 - self.alpha) * kl_div.item()
            self.running_motion = self.alpha * self.running_motion + (1 - self.alpha) * motion_loss.item()

        norm_recon = recon_loss / (self.running_recon + 1e-8)
        norm_kl = kl_div / (self.running_kl + 1e-8)
        norm_motion = motion_loss / (self.running_motion + 1e-8)

        if self.use_log:
            norm_recon = torch.log1p(torch.clamp(norm_recon, min=0))
            norm_kl = torch.log1p(torch.clamp(norm_kl, min=0))
            norm_motion = torch.log1p(torch.clamp(norm_motion, min=0))
        if self.adaptive:
            total_mag = norm_recon + norm_kl + norm_motion + 1e-8
            w_recon = 1 - (norm_recon / total_mag)
            w_kl = 1 - (norm_kl / total_mag)
            w_motion = 1 - (norm_motion / total_mag)
        else:
            w_recon = w_kl = w_motion = 1.0

        weighted_recon = self.recon_weight * norm_recon
        weighted_kl = self.beta * norm_kl
        weighted_motion = self.motion_weight * norm_motion

        total_loss = self.scale * (w_recon*weighted_recon + w_kl*weighted_kl + w_motion*weighted_motion)
        return total_loss, weighted_recon, weighted_kl, weighted_motion

