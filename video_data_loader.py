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
            for i in range(0, frame_count - self.skip_frames * self.sequence_length, self.skip_frames):
                if i + self.skip_frames * (self.sequence_length - 1) < frame_count:
                    frame_indices.append((video_idx, i))
        return frame_indices
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        '''
        Now loads and processes frames on-demand when requested. 
        This means that only the frames needed for the current batch are loaded into memory.
        Optical flow is computed on-the-fly for each batch, 
        which can be more memory-efficient but might be slightly slower during training.
        '''
        video_idx, start_frame = self.frame_indices[idx]
        video_path = self.video_paths[video_idx]
        
        frames = []
        flows = []
        
        cap = cv2.VideoCapture(video_path)
        try:
            for i in range(self.sequence_length):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * self.skip_frames)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Could not read frame {start_frame + i * self.skip_frames} from {video_path}")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
                
                if i > 0:
                    prev_frame = cv2.cvtColor(frames[-2].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
                    curr_frame = cv2.cvtColor(frames[-1].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()
                    flows.append(flow)
        finally:
            cap.release()
        
        frames_tensor = torch.stack(frames)
        flows_tensor = torch.stack(flows)
        
        return frames_tensor[:-1], flows_tensor, frames_tensor[-1]


class MemoryEfficientDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        self.collate_fn = self._collate_fn

    @staticmethod
    def _collate_fn(batch):
        # a custom function to properly stack the batch items
        frames, flows, targets = zip(*batch)
        return torch.stack(frames), torch.stack(flows), torch.stack(targets)

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

