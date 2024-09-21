import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from contextlib import contextmanager


class MemoryEfficientVideoDataset(Dataset):
    '''
    Only stores the video paths and frame indices, 
    rather than loading all the frames into memory at once.
    Only the frames needed for the current batch are loaded into memory, 
    significantly reducing the overall memory footprint.
    The dataset initialization is much quicker,
    since it only involves indexing the frames rather than loading all the data.
    Allows larger datasets that might not fit entirely in memory.
    Keep in mind that this approach may increase the I/O operations during training, 
    which could potentially slow down the training process if your storage is not fast enough. 
    However, for most systems, 
    the memory savings should outweigh the potential speed decrease.
    '''
    def __init__(self, video_paths, sequence_length=5, target_size=(144, 256), skip_frames=1):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.skip_frames = skip_frames
        self.frame_indices = self._index_frames()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    
    def _index_frames(self):
        frame_indices = []
        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            max_t = frame_count - self.skip_frames  # Ensure target frame exists
            for t in range((self.sequence_length - 1) * self.skip_frames, max_t):
                frame_indices.append((video_idx, t))
        return frame_indices

    
    def __len__(self):
        return len(self.frame_indices)
    

    def __getitem__(self, idx):
        video_idx, end_frame = self.frame_indices[idx]
        video_path = self.video_paths[video_idx]
        
        # Calculate the time steps for input frames
        frame_times = [end_frame - i * self.skip_frames for i in reversed(range(self.sequence_length))]
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        try:
            # Load input frames at specified times
            for ft in frame_times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ft)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Could not read frame {ft} from {video_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            
            # Load target frame at time t + skip_frames
            target_frame_idx = end_frame + self.skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, target_frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read target frame {target_frame_idx} from {video_path}")
            target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
            target_frame = self.transform(target_frame)
        finally:
            cap.release()
        
        # Compute flows between consecutive input frames
        flows = []
        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            next_frame = frames[i + 1]
            prev_frame_gray = cv2.cvtColor(prev_frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            next_frame_gray = cv2.cvtColor(next_frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()
            flows.append(flow)
        
        # Convert lists to tensors
        frames_tensor = torch.stack(frames)        # Shape: (sequence_length, C, H, W)
        flows_tensor = torch.stack(flows)          # Shape: (sequence_length - 1, 2, H, W)
        
        return frames_tensor, flows_tensor, target_frame


class MemoryEfficientDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        self.collate_fn = self._collate_fn

    @staticmethod
    def _collate_fn(batch):
        frames, flows, targets = zip(*batch)
        frames = torch.stack(frames)
        flows = torch.stack(flows)
        targets = torch.stack(targets)
        return frames, flows, targets

    def __del__(self):
        # Ensure that the workers are shut down properly
        self._iterator = None
        if hasattr(self, '_workers'):
            for w in self._workers:
                w.terminate()
            for w in self._workers:
                w.join(timeout=5)


@contextmanager
def load_video_data(video_paths, batch_size, sequence_length, target_size, skip_frames, num_workers):
    dataset = MemoryEfficientVideoDataset(video_paths, sequence_length, target_size, skip_frames)
    dataloader = MemoryEfficientDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    try:
        yield dataloader
    finally:
        del dataloader  # This will trigger the __del__ method, ensuring proper cleanup



if __name__ == "__main__":
    
    video_paths = ["data/bounce 1-1.mkv", "data/fall 1-4.mkv"]
    batch_size = 2
    sequence_length = 5
    target_size = (144, 256)
    skip_frames = 1
    num_workers = 1
    
    dataset = MemoryEfficientVideoDataset(video_paths, sequence_length, target_size, skip_frames)
    frames_tensor, flows_tensor, target_frame = dataset[0]

    print("Input Frames Shape:", frames_tensor.shape)        # Should be (sequence_length, C, H, W)
    print("Flows Shape:", flows_tensor.shape)                # Should be (sequence_length - 1, 2, H, W)
    print("Target Frame Shape:", target_frame.shape)         # Should be (C, H, W)
    

    with load_video_data(video_paths, batch_size, sequence_length, target_size, skip_frames, num_workers) as dataloader:
        for frames, flows, targets in dataloader:
            print(frames.shape, flows.shape, targets.shape)
            break