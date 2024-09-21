import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class EnhancedConvBetaVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, sequence_length=5, 
                frame_height=144, frame_width=256, base_conv_size=8):
        super(EnhancedConvBetaVAE, self).__init__()
        
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.base_conv_size = base_conv_size
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.base_conv_size*(2**0), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**0), self.base_conv_size*(2**1), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**1), self.base_conv_size*(2**2), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**2), self.base_conv_size*(2**3), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Optical flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, self.base_conv_size*(2**0), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**0), self.base_conv_size*(2**1), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**1), self.base_conv_size*(2**2), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_conv_size*(2**2), self.base_conv_size*(2**3), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(self.base_conv_size*(2**4), self.base_conv_size*(2**4), kernel_size=3)
        
        # Calculate the size of the flattened features
        with torch.no_grad():
            sample_frame = torch.zeros(1, input_channels, self.frame_height, self.frame_width)
            sample_flow = torch.zeros(1, 2, self.frame_height, self.frame_width)
            frame_features = self.frame_encoder(sample_frame)
            flow_features = self.flow_encoder(sample_flow)
            combined_features = torch.cat([frame_features, flow_features], dim=1)
            h, c = self.init_hidden(1, combined_features.shape[2:])
            h, _ = self.conv_lstm(combined_features, (h, c))
            self.flattened_size = h.numel()
            self.feature_size = h.shape[1:]
        
        # Fully connected layers
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_conv_size*(2**4), self.base_conv_size*(2**3), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.base_conv_size*(2**3), self.base_conv_size*(2**2), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.base_conv_size*(2**2), self.base_conv_size*(2**1), kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.base_conv_size*(2**1), input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        
    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.base_conv_size*(2**4), height, width),
                torch.zeros(batch_size, self.base_conv_size*(2**4), height, width))
    
    def encode(self, frames, flows):
        batch_size = frames.size(0) # [batch_size, seq_len, channels, height, width]
        seq_len = frames.size(1)
        
        # Initial hidden state
        frame_features = self.frame_encoder(frames[:, 0])
        flow_features = self.flow_encoder(flows[:, 0])
        combined_features = torch.cat([frame_features, flow_features], dim=1)
        h, c = self.init_hidden(batch_size, combined_features.shape[2:])
        h, c = self.conv_lstm(combined_features, (h, c))
        
        # Loop over the rest of the sequence
        for t in range(1, seq_len - 1):
            frame_features = self.frame_encoder(frames[:, t])
            flow_features = self.flow_encoder(flows[:, t])
            combined_features = torch.cat([frame_features, flow_features], dim=1)
            h, c = self.conv_lstm(combined_features, (h, c))
        
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(h.size(0), *self.feature_size)
        return self.decoder(h)

    def forward(self, frames, flows):
        # frames size: (batch_size, sequence_length-1, channels, height, width)
        # flows size: (batch_size, sequence_length-1, 2, height, width)
        mu, logvar = self.encode(frames, flows)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
