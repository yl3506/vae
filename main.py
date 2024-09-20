import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from video_data_loader import load_video_data
from beta_vae_model import EnhancedConvBetaVAE
import random
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import shutil
import numpy as np
import resource
import sys
from matplotlib.colors import Normalize


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


def calculate_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def kl_divergence(mu, logvar):
    mu = torch.clamp(mu, -10, 10) # Clamp values to prevent extreme outputs
    logvar = torch.clamp(logvar, -10, 10)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


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
    # Compute reconstruction loss
    recon_loss = F.mse_loss(recon_frame, target_frame, reduction='none')
    
    # Compute motion mask (difference between frames)
    motion_threshold = 0.01
    motion_mask = torch.abs(target_frame - orig_frame).mean(dim=1, keepdim=True)
    motion_mask = (motion_mask > motion_threshold).float() # thresholding to reduce noise sensitivity
    
    # Tile the motion mask to match the number of channels in the frame
    motion_mask = motion_mask.repeat(1, 3, 1, 1)

    # Combine segmentation and motion masks
    combined_mask = emphasis*motion_mask
    
    # Apply motion masks to loss
    weighted_motion_loss = (recon_loss*emphasis*motion_mask).sum()
    
    return weighted_motion_loss


def gradient_difference_loss(pred_frame, target_frame):
    '''
    focuses on the consistency of gradients between the predicted and target frames, 
    which can capture motion and structural information
    Edge Detection:
        Gradients effectively highlight edges and texture in an image.
        By comparing gradients, 
        we're focusing on the structural similarities between frames rather than just pixel-wise differences.
    Invariance to Uniform Changes:
        This loss is less sensitive to uniform changes in brightness or color across the entire image.
        It cares more about preserving the relative changes between neighboring pixels.
    Sharpness Preservation:
        By explicitly comparing gradients, 
        this loss encourages the prediction of sharper, more detailed images.
        It penalizes blurry predictions more heavily than a simple pixel-wise loss would.
    Direction Sensitivity:
        By separating x and y gradients, 
        the loss is sensitive to the direction of edges and textures.
        This can help in preserving directional structures like lines or oriented textures.
    L1 vs L2 Norm:
        The use of L1 loss (absolute difference) rather than L2 (squared difference) 
        makes the loss less sensitive to large gradient differences.
        This can be beneficial in preserving sharp edges without over-penalizing large differences.
    '''
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = gradient(pred_frame)
    target_dx, target_dy = gradient(target_frame)

    loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    return loss


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


def save_checkpoint(model, optimizer, epoch, losses, best_loss, save_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'best_loss': best_loss
    }
    
    filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pkl')
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved successfully: {filename}")
        
        if is_best:
            best_filename = os.path.join(save_dir, 'best_model.pkl')
            with open(best_filename, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"Best model saved successfully: {best_filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        raise


def load_checkpoint(model, optimizer, filename):
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['losses'], checkpoint['best_loss']
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise


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


def main():

    # Hyperparameters
    # get current directory
    root_dir = os.getcwd() + "/"
    print(f"Current directory: {root_dir}")
    video_dir = root_dir + "data/"
    resume_from_pkl = f"{root_dir}logs/20240821-161859/checkpoints/best_model.pkl" 
    # resume_from_pkl = None
    target_size = (144, 256) # size of video frame (h, w)
    skip_frames = 3 # interval of frame prediction

    batch_size = 32
    num_videos_in_epoch = 10 # number of sample videos in epoch
    num_epochs = 100
    num_workers = 2
    log_frequency = 5 # plot results every x epochs
    save_frequency = 20  # save model every x epochs
    max_retries = 3 # number of retries of checkpoint saving

    latent_dim = 256 # for fc between encoder and decoder
    base_conv_size = 8 # base size of conv net hidden channels
    sequence_length = 5 # prev num of frames as context
    learning_rate = 1e-3
    
    # loss_scale = 1e5 # used in BalancedLoss, scale the total loss for numerical stability
    recon_weight = 1 # weight (relative to 1) for the reconstruction loss
    beta = 3 # weight (relative to 1) for the kl divergence in the loss function
    motion_weight = 2 # weight (relative to 1) for motion error in the loss function
    motion_loss_scale = 1e1#1e6 # scaling the motion loss in the loss calculation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    check_disk_space("/")  # Check disk space of the root directory
    increase_file_limit() # increase num open files allowed

    # Create directories for saving results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"{root_dir}logs/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not os.access(checkpoint_dir, os.W_OK):
        raise PermissionError(f"No write permission for directory: {checkpoint_dir}")
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # paths to all videos
    all_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mkv')]

    # Lists to store loss values
    losses = {'total': [], 'recon': [], 'kl': [], 'motion': [], } #'norm_recon': [], 'norm_kl': [], 'norm_motion': []}
    best_loss = float('inf')
    start_epoch = 0

    try:
        # Initialize model
        model = EnhancedConvBetaVAE(input_channels=3, 
                                    latent_dim=latent_dim, 
                                    sequence_length=sequence_length,
                                    frame_height=target_size[0],
                                    frame_width=target_size[1],
                                    base_conv_size=base_conv_size)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # balanced_loss = BalancedLoss(recon_weight=recon_weight, beta=beta, motion_weight=motion_weight, scale=loss_scale, use_log=True, adaptive=True)
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Model state dict size: {sum(param.numel() * param.element_size() for param in model.state_dict().values()) / (1024 * 1024):.2f} MB")

        # Check if there's a checkpoint to resume from in the cur dir
        latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')], default=None)
        if latest_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            start_epoch, losses, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
            print(f"Resuming training from epoch {start_epoch}")
        
        # manually load checkpoint if provided
        if resume_from_pkl != None:
            start_epoch, losses, _ = load_checkpoint(model, optimizer, resume_from_pkl)
            print(f"Loading checkpoint from epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, start_epoch+num_epochs):

            # select a subset of videos in the full dataset for this epoch
            subset_videos = random.sample(all_videos, min(num_videos_in_epoch, len(all_videos)))
            # Use the context manager for the dataloader
            with load_video_data(video_paths=subset_videos, batch_size=batch_size, sequence_length=sequence_length, 
                    target_size=target_size, skip_frames=skip_frames, num_workers=num_workers) as dataloader:
                
                # Check model output
                for sample_batch in dataloader: 
                    check_model_output(model, sample_batch, device)
                    break

                model.train()
                
                # Progress bar for batches
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}", leave=False)
                epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'motion': 0, } #'norm_recon': 0, 'norm_kl': 0, 'norm_motion': 0}

                # start the batch
                for batch_idx, (frames, flows, target_frame) in enumerate(pbar):
                    frames, flows, target_frame = frames.to(device), flows.to(device), target_frame.to(device)
                    
                    optimizer.zero_grad()
                    # recon_frame, mu, logvar, segmentation_mask = model(frames, flows)
                    recon_frame, mu, logvar = model(frames, flows)
                    
                    # recon_loss = segmentation_aware_loss(recon_frame, frames[:,-1], target_frame, segmentation_mask)
                    recon_loss = F.mse_loss(recon_frame, target_frame, reduction='sum')
                    kl_div = kl_divergence(mu, logvar)
                    # motion_loss = accumulated_motion_loss(recon_frame, target_frame, frames)
                    motion_loss = emphasized_motion_loss(recon_frame, frames[:,-1], target_frame, emphasis=1)

                    total_loss = recon_loss + beta*kl_div + motion_weight*motion_loss_scale*motion_loss
                    if torch.isnan(total_loss):
                        print(f"NaN loss detected at batch {batch_idx}. Skipping update.")
                        continue

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                    optimizer.step()
                    
                    epoch_losses['total'] += total_loss.item()
                    epoch_losses['recon'] += recon_loss.item()
                    epoch_losses['kl'] += kl_div.item()
                    epoch_losses['motion'] += motion_loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'kl': f"{kl_div.item():.4f}",
                        'motion': f"{motion_loss.item():.6f}",
                    })
                
                # Calculate average losses
                for key in epoch_losses:
                    epoch_losses[key] /= len(dataloader.dataset)
                    losses[key].append(epoch_losses[key])
                
                # Update epoch progress bar
                tqdm.write(f"Epoch {epoch+1}/{start_epoch+num_epochs}, "
                           f"Loss: {epoch_losses['total']:.4f}, "
                           f"Recon Loss: {epoch_losses['recon']:.4f}, "
                           f"KL Div: {epoch_losses['kl']:.4f}, "
                           f"Motion Loss: {epoch_losses['motion']:.6f}, "
                           )
            
            # outside dataloader context, space freed
            is_best = epoch_losses['total'] < best_loss
            best_loss = min(epoch_losses['total'], best_loss)
            if (epoch + 0) % save_frequency == 0 or is_best:
                # Save checkpoint with retry mechanism
                for retry in range(max_retries):
                    try:
                        save_checkpoint(model, optimizer, epoch + 1, losses, best_loss, checkpoint_dir, is_best)
                        break  # If successful, break out of the retry loop
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"Error saving checkpoint (attempt {retry + 1}/{max_retries}): {str(e)}")
                            time.sleep(20)  # Wait a bit before retrying
                        else:
                            print(f"Failed to save checkpoint after {max_retries} attempts.")
                            raise  # Re-raise the last exception if all retries fail
            
            # Save loss history
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(losses, f)
            
            # Visualize losses
            if (epoch + 0) % log_frequency == 0:
                visualize_loss(losses['total'], losses['recon'], losses['kl'], losses['motion'],
                                os.path.join(vis_dir, f'loss_plot.png'))
            
            # Visualize reconstructions for the batch
            if (epoch + 0) % log_frequency == 0:
                # Create a temporary dataloader for visualization
                with load_video_data(video_paths=subset_videos, batch_size=batch_size, sequence_length=sequence_length, 
                                     target_size=target_size, skip_frames=skip_frames, num_workers=1) as vis_dataloader:
                    model.eval()
                    with torch.no_grad():
                        sample_frames, sample_flows, sample_target = next(iter(vis_dataloader))
                        sample_frames, sample_flows, sample_target = sample_frames.to(device), sample_flows.to(device), sample_target.to(device)
                        # sample_recon, _, _, sample_mask = model(sample_frames, sample_flows)
                        sample_recon, _, _ = model(sample_frames, sample_flows)
                        visualize_frames(sample_frames[:, -1], None, f"Original Frames (Epoch {epoch+1})", 
                                         os.path.join(vis_dir, f'epoch_{epoch+1}_frames_original.png'))
                        visualize_frames(sample_target, None, f"Target Next Frames (Epoch {epoch+1})", 
                                         os.path.join(vis_dir, f'epoch_{epoch+1}_frames_target.png'))
                        # visualize_frames(sample_recon, sample_mask, f"Predicted Next Frames (Epoch {epoch+1})", 
                        #                  os.path.join(vis_dir, f'epoch_{epoch+1}_frames_predicted.png'))
                        visualize_frames(sample_recon, None, f"Predicted Next Frames (Epoch {epoch+1})", 
                                         os.path.join(vis_dir, f'epoch_{epoch+1}_frames_predicted.png'))

        print("Training completed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

