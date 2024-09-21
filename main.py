import os
import torch
from video_data_loader import load_video_data
from beta_vae_model import EnhancedConvBetaVAE
import random
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
import utils


def train():

    # Lists to store loss values
    losses = {'total': [], 'recon': [], 'kl': [], 'motion': [], } #'norm_recon': [], 'norm_kl': [], 'norm_motion': []}
    best_loss = float('inf')
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, start_epoch+num_epochs):

        # select a subset of videos in the full dataset for this epoch
        subset_videos = random.sample(all_videos, min(num_videos_in_epoch, len(all_videos)))
        # Use the context manager for the dataloader
        with load_video_data(video_paths=subset_videos, batch_size=batch_size, sequence_length=sequence_length, 
                target_size=target_size, skip_frames=skip_frames, num_workers=num_workers) as dataloader:
            
            # Check model output
            for sample_batch in dataloader: 
                utils.check_model_output(model, sample_batch, device)
                break

            model.train()
            
            # Progress bar for batches
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}", leave=False)
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'motion': 0, } #'norm_recon': 0, 'norm_kl': 0, 'norm_motion': 0}
            num_samples = 0  # Keep track of the total number of samples processed

            # start the batch
            for batch_idx, (frames, flows, target_frame) in enumerate(pbar):
                frames, flows, target_frame = frames.to(device), flows.to(device), target_frame.to(device)
                bs = frames.size(0)
                num_samples += bs

                optimizer.zero_grad()
                recon_frame, mu, logvar = model(frames, flows)
                
                recon_loss = F.mse_loss(recon_frame, target_frame, reduction='mean')
                kl_div = utils.kl_divergence(mu, logvar)
                orig_frame = frames[:, -1]  # Frame at time t
                motion_loss = utils.emphasized_motion_loss(recon_frame, orig_frame, target_frame, emphasis=1)

                total_loss = recon_loss + beta * kl_div + motion_weight * motion_loss_scale * motion_loss

                if torch.isnan(total_loss):
                    print(f"NaN loss detected at batch {batch_idx}. Skipping update.")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_losses['total'] += total_loss.item() * frames.size(0)  # Multiply by batch size
                epoch_losses['recon'] += recon_loss.item() * frames.size(0)
                epoch_losses['kl'] += kl_div.item() * frames.size(0)
                epoch_losses['motion'] += motion_loss.item() * frames.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'kl': f"{kl_div.item():.4f}",
                    'motion': f"{motion_loss.item():.6f}",
                })
                
            # Calculate average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_samples
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
                    utils.save_checkpoint(model, optimizer, epoch + 1, losses, best_loss, checkpoint_dir, is_best)
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
            utils.visualize_loss(losses['total'], losses['recon'], losses['kl'], losses['motion'],
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
                    sample_recon, _, _ = model(sample_frames, sample_flows)
                    utils.visualize_frames(sample_frames[:, -1], None, f"Original Frames (Epoch {epoch+1})", 
                                        os.path.join(vis_dir, f'epoch_{epoch+1}_frames_original.png'))
                    utils.visualize_frames(sample_target, None, f"Target Next Frames (Epoch {epoch+1})", 
                                        os.path.join(vis_dir, f'epoch_{epoch+1}_frames_target.png'))
                    utils.visualize_frames(sample_recon, None, f"Predicted Next Frames (Epoch {epoch+1})", 
                                        os.path.join(vis_dir, f'epoch_{epoch+1}_frames_predicted.png'))
        
    print("Training completed.")


if __name__ == "__main__":
    
    # Hyperparameters
    root_dir = os.getcwd() + "/"
    print(f"Current directory: {root_dir}")
    video_dir = root_dir + "data/"
    resume_from_checkpoint = f"{root_dir}logs/20240821-161859/checkpoints/best_model.pkl" 
    resume_from_checkpoint = None
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
    motion_loss_scale = 1 #1e1 # scaling the motion loss in the loss calculation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    utils.check_disk_space("/")  # Check disk space of the root directory
    utils.increase_file_limit() # increase num open files allowed

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
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=utils.get_epoch_number)
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        start_epoch, losses, best_loss = utils.load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
        
    # manually load checkpoint if provided
    if resume_from_checkpoint is not None:
        start_epoch, losses, best_loss = utils.load_checkpoint(model, optimizer, resume_from_checkpoint)
        print(f"Loading checkpoint from epoch {start_epoch}")
    elif checkpoint_files:
        # Load the latest checkpoint from the current directory
        latest_checkpoint = max(checkpoint_files, key=utils.get_epoch_number)
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        start_epoch, losses, best_loss = utils.load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")


    train()