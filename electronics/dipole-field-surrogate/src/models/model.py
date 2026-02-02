import torch.nn as nn
import math
import torch

class DipoleFieldModel(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_channels=64,
        latent_h=8,
        latent_w=8,
        output_channels=2,
        output_h=32,
        output_w=32,
    ):
        """
        Flexible encoder-decoder model for dipole field generation.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        latent_channels : int
            Number of channels in latent representation
        latent_h : int
            Height of latent spatial map
        latent_w : int
            Width of latent spatial map
        output_channels : int
            Number of output channels (e.g., 2 for x,y components)
        output_h : int
            Target output height (default: 32)
        output_w : int
            Target output width (default: 32)
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.output_h = output_h
        self.output_w = output_w
        self.latent_dimension = latent_channels * latent_h * latent_w

        # Encoder: input_dim -> latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dimension),
        )

        # Build dynamic decoder
        self.decoder = self._build_decoder(
            latent_channels, 
            latent_h, 
            latent_w,
            output_channels,
            output_h,
            output_w
        )

    def _build_decoder(self, in_channels, in_h, in_w, out_channels, out_h, out_w):
        """
        Dynamically build decoder to upsample from (in_h, in_w) to (out_h, out_w).
        """
        layers = []
        current_h, current_w = in_h, in_w
        current_channels = in_channels
        
        # Calculate number of upsampling layers needed
        target_scale_h = out_h / in_h
        target_scale_w = out_w / in_w
        
        # Determine number of 2x upsampling layers
        num_upsamples = int(math.log2(min(target_scale_h, target_scale_w)))
        
        # Progressive upsampling with decreasing channels
        channel_progression = [in_channels, 64, 32, 16]
        
        for i in range(num_upsamples):
            next_channels = channel_progression[min(i + 1, len(channel_progression) - 1)]
            
            layers.extend([
                nn.ConvTranspose2d(
                    current_channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
            ])
            
            current_channels = next_channels
            current_h *= 2
            current_w *= 2
        
        # Final adjustment layer if needed
        if current_h != out_h or current_w != out_w:
            # Use adaptive upsampling + conv to reach exact target size
            layers.extend([
                nn.Upsample(size=(out_h, out_w), mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
        
        # Output layer
        layers.append(
            nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input features
        
        Returns:
        --------
        output : torch.Tensor, shape (batch_size, output_channels, output_h, output_w)
            Generated field
        """
        batch_size = x.size(0)

        # Encode to latent representation
        latent = self.encoder(x)  # (batch_size, latent_dimension)

        # Reshape to spatial format
        latent = latent.view(
            batch_size,
            self.latent_channels,
            self.latent_h,
            self.latent_w,
        )  # (batch_size, latent_channels, latent_h, latent_w)

        # Decode to output field
        output = self.decoder(latent)  # (batch_size, output_channels, output_h, output_w)

        return output






# class DipoleFieldModel(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         latent_channels=64,
#         latent_h=8,
#         latent_w=8,
#         output_channels=2,
#         ):
    
#         super().__init__()

#         self.latent_channels = latent_channels
#         self.latent_h = latent_h
#         self.latent_w = latent_w
#         self.latent_dimension = latent_channels * latent_h * latent_w

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.latent_dimension),
#         )

#         self.decoder = nn.Sequential(

#             # (C, 8, 8) -> (32, 16, 16)
#             nn.ConvTranspose2d(
#                 self.latent_channels,
#                 32,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             # (32, 16, 16) -> (16, 32, 32)
#             nn.ConvTranspose2d(
#                 32,
#                 16,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             # (16, 32, 32) -> (output_channels, 32, 32)
#             nn.Conv2d(
#                 16,
#                 output_channels,
#                 kernel_size=3,
#                 padding=1,
#             ),
#         )

#     def forward(self, x):
#         # x  : (batch_size, input_dim)
#         batch_size = x.size(0)

#         # Encode
#         latent = self.encoder(x)

#         # Reshape to (batch_size, C, H, W)
#         latent = latent.view(
#             batch_size,
#             self.latent_channels,
#             self.latent_h,
#             self.latent_w,
#         )

#         # Decode to field
#         output = self.decoder(latent)

#         return output  # (batch_size, output_channels, 32, 32)

