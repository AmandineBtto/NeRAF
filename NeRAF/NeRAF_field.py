"""
NeRAF neural acoustic field (NAcF)

"""

from typing import Literal, Optional

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
import torch.nn as nn
import torch.nn.functional as F
import torch
from nerfstudio.field_components.encodings import NeRFEncoding

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

class NeRAFVisionFieldValue(nn.Module): # Should be deleted -> NOT TESTED

    def __init__(self,module):
        super().__init__()
        self.module = module

    def forward(self,ray_samples, compute_normals=False):
        return self.module(ray_samples, compute_normals=compute_normals)
    

class NeRAFAudioSoundField(nn.Module):

    def __init__(self,in_size,W,sound_rez=2,N_frequencies=257):
        super().__init__()
        self.soundfield = nn.ModuleList(
            [nn.Linear(in_size, 5096), nn.Linear(5096, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 1024), nn.Linear(1024, W)])
            
        self.STFT_linear = nn.ModuleList(
            [nn.Linear(W, N_frequencies) for _ in range(sound_rez)])
        
    def forward(self, h):

        for i, layer in enumerate(self.soundfield):
            h = layer(h)
            h = F.leaky_relu(h, negative_slope=0.1)

        output = []
        feat = h

        for i, layer in enumerate(self.STFT_linear):
            h = layer(feat)
            h = F.tanh(h)*10
            # h = F.leaky_relu(h, negative_slope=0.1) # if mag stft prediction
            mono = h.unsqueeze(1) 
            output.append(mono)

        output = torch.cat(output, dim=1) 

        return output
        