"""
NeRAF model
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from .NeRAF_field import NeRAFVisionFieldValue, NeRAFAudioSoundField
from .NeRAF_resnet3d import ResNet3D_helper
from .NeRAF_evaluator import RAFEvaluator, SoundSpacesEvaluator, STFTLoss

from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
import torch
import numpy as np

from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.cameras.rays import RaySamples, Frustums

import torch.nn.functional as F
import math

from nerfstudio.model_components.losses import (
    MSELoss
)

from nerfstudio.data.scene_box import SceneBox
from torchaudio.transforms import GriffinLim

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nerfstudio.viewer.viewer_elements import *
from scipy.spatial.transform import Rotation as R
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

import librosa 
from scipy.signal import fftconvolve

### For custom viewer button
from scipy.io import wavfile

@dataclass
class NeRAFVisionModelConfig(NerfactoModelConfig):
    """NeRAF Vision Model Config."""

    _target: Type = field(default_factory=lambda: NeRAFVisionModel)

class NeRAFVisionModel(NerfactoModel):
    """NeRAF Vision Model"""

    config: NeRAFVisionModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.field = NeRAFVisionFieldValue(self.field) # TODO Test if we can remove this line
        self.audio_model = None
    
    """ Merge vision and audio outputs for viewer """
    def get_outputs(self, ray_bundle):
        output = super().get_outputs(ray_bundle)
        output["rgb"] = torch.clip(output["rgb"], 0, 1) 
        return output

    def get_outputs_for_camera(self, camera, obb_box, eval=False):
        if not eval: 
            output_audio = self.audio_model.get_outputs_for_camera(camera, obb_box)
        output_vision = super().get_outputs_for_camera(camera, obb_box)
        output_vision["rgb"] = torch.clip(output_vision["rgb"], 0, 1)
        if not eval:
            output = {**output_audio, **output_vision}
        else: 
            output = output_vision
        return output


@dataclass
class NeRAFAudioModelConfig(ModelConfig):
    """NeRAF Audio Model Config."""

    _target: Type = field(default_factory=lambda: NeRAFAudioModel)

    # default paramters
    dataset: str = 'SoundSpaces'
    use_grid: bool = True
    grid_step: float = 1/128
    N_features: int = 1024
    use_multiple_viewing_directions: bool = True
    loss_factor: float = 1e-3
    max_len: int = 76
    W_field: int = 512
    fs: int = 22050
    criterion: str = 'SC+SLMSE'
    N_freq_stft: int = 257
    hop_len: int = 128
    win_len: int = 512


class NeRAFAudioModel(Model):
    """NeRAF Audio Model."""

    config: NeRAFAudioModelConfig

    def default_RAF_config(self):
        self.config.fs = 48000
        self.config.max_len = 0.32
        if self.config.fs == 48000:
            self.config.N_freq_stft = 513
            self.config.hop_len = 256
            self.config.win_len = 512
        elif self.config.fs == 16000:
            self.config.N_freq_stft = 257
            self.config.hop_len = 128
            self.config.win_len= 256

    def populate_modules(self):
        super().populate_modules()

        self.dataset = self.config.dataset

        if self.dataset == 'RAF':
            self.default_RAF_config()
            self.max_len = int(self.config.max_len * self.config.fs) // self.config.hop_len
            self.mic_ch = 1
            self.evaluator = RAFEvaluator(fs=self.config.fs)
        else:
            self.max_len = self.config.max_len
            self.mic_ch = 2
            self.evaluator = SoundSpacesEvaluator(fs=self.config.fs)

        self.use_grid = self.config.use_grid
        self.loss_factor = self.config.loss_factor
        self.criterion_name = self.config.criterion
        self.istft_transform = GriffinLim(n_fft=(self.config.N_freq_stft-1)*2, win_length = self.config.win_len, hop_length=self.config.hop_len, power = 1)

        if self.criterion_name == 'MSE':
            print('Using MSE loss')
            self.criterion = torch.nn.MSELoss(reduction='mean')        
        else:
            if 'MSE' in self.criterion_name:
                print('Using MSE SL loss')
            else: 
                print('Using L1 SL loss')
            self.criterion = STFTLoss(loss_type='mse' if 'MSE' in self.criterion_name else 'l1')

        

        self.sampler_uniform = UniformSampler(num_samples=64) #not used for audio -> None (TO test)

        self.spatial_distortion = None # will be retrieved from vision model

        # Encoding 
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding=SHEncoding(
            levels=4,
            implementation="tcnn",
        )

        self.input_ch_time = self.time_encoding.get_out_dim()
        self.input_ch_pose = self.position_encoding.get_out_dim()
        self.input_ch_rot = self.rot_encoding.get_out_dim()

        print('Audio scene box', self.scene_box.aabb)

        W=self.config.W_field
        N_freq = self.config.N_freq_stft

        if self.use_grid:
            self.grid_size = np.array([0,1,0,1,0,1])
            self.grid_step = self.config.grid_step
            self.N_features = self.config.N_features    
            self.reset_grid(device="cpu")
            print("Grid shape", self.grid.shape, "N_features", self.N_features)
            
            self.resnet3d = ResNet3D_helper(in_channels=7, backbone='resnet50', pretrained=False, grid_step=self.grid_step, N_features=self.N_features)
            out_shape = self.resnet3d.backbone_net(torch.zeros(self.grid.shape).unsqueeze(0))[-1].shape
            print("Resnet output shape", out_shape)
            self.grid_size_after_resnet = out_shape[-1] * out_shape[-2] * out_shape[-3] * out_shape[-4]
            self.field = NeRAFAudioSoundField(int(self.grid_size_after_resnet+self.input_ch_time+self.input_ch_pose*2+self.input_ch_rot),W,sound_rez=self.mic_ch, N_frequencies=N_freq)
            
            self._delta = 1e-2

            if self.config.use_multiple_viewing_directions:
                self.view_dirs = self._generate_fixed_viewing_directions()
            else:
                self.view_dirs = None

            grid_size = self.grid_size
            grid_step = self.grid_step
            grid_coordinates = torch.meshgrid(torch.arange(grid_size[0]+grid_step/2, grid_size[1], grid_step), torch.arange(grid_size[2]+grid_step/2, grid_size[3], grid_step), torch.arange(grid_size[4]+grid_step/2, grid_size[5], grid_step), indexing='ij')
            grid_coordinates = torch.stack(grid_coordinates, dim=-1)
            self.coordinates_to_render = grid_coordinates.view(-1, 3)
            self.grid_batch_i = 0
            self.grid = None # To force reset first time
        
        else:
            self.field = NeRAFAudioSoundField(int(self.input_ch_time+self.input_ch_pose*2+self.input_ch_rot),W,sound_rez=self.mic_ch, N_frequencies=N_freq)
        
        # for viewer purposes -> to be changed later
        self.eval_source_pose = None
        self.eval_mic_pose = None
        self.eval_rot = None
        self.eval_gt = None

        self.source_coordinates_gui = ViewerVec3(name="Source pos (relative)", default_value=(0, 0, 0))
        self.play_btn = ViewerButton(name="Save sound", cb_hook=self.handle_btn)
        self.wav_path = ViewerText(name="Sound input", default_value='./path')
        self.wav_path_output = ViewerText(name="Sound output", default_value='./path')
        self.viewer_control = ViewerControl()

    def handle_btn(self,handle):
        # WIP: viewer support (not finished yet)
        print('Warning: WIP Not finished yet')
        camera = self.viewer_control.get_camera(100,100)
        output = self.get_outputs_for_camera(camera, None,0)
        stft = output["raw_output"].permute(1,2,0) #shape [C, 257, T]
            
        mag_prd = np.clip(np.exp(stft.cpu().detach().numpy()) - 1e-3, 0.0, 10000.00)
        wav_prd = self.istft_transform(torch.from_numpy(mag_prd).cuda()).cpu().numpy()
        wav_prd = np.clip(wav_prd, -1.0, 1.0)

        input_wav, sample_rate_loaded = librosa.load(self.wav_path.value, sr=self.config.fs, mono=True)

        # using wavfile ----
        # loaded = wavfile.read(self.wav_path.value)
        # input_wav = loaded[1]

        # if(input_wav.dtype == np.int16):
        #     input_wav = input_wav.astype(np.float32)/32767.0
        # elif(input_wav.dtype == np.int32):
        #     input_wav = input_wav.astype(np.float32)/2147483647.0
        # elif(input_wav.dtype == np.uint8):
        #     input_wav = input_wav.astype(np.float32)/255.0
        # elif(input_wav.dtype == np.float32):
        #     input_wav = input_wav
        # else:
        #     print("Unknown type", input_wav.dtype)
        #     return
        
        #if input is stereo (automatic conversion to mono with librosa)
        # if(input_wav.ndim == 2):
        #     input_wav = (input_wav[:,0] + input_wav[:,1])/2.0
        # --- 
        
        #if more than 5 seconds
        if(input_wav.shape[0] > 5*sample_rate_loaded):
            input_wav = input_wav[:5*sample_rate_loaded]

        # auralization
        out_0=fftconvolve(input_wav,wav_prd[0,:])
        out_1=fftconvolve(input_wav,wav_prd[1,:])

        # stack the two channels
        output_wav = output_wav.astype(np.float32)
        
        wavfile.write(self.wav_path_output.value, sample_rate_loaded, output_wav.T)
        print("Sound saved at", self.wav_path_output.value)
  
    def reset_grid(self,device=None):
        device = self.device if device is None else device
        self.grid = torch.zeros((7, int((self.grid_size[1] - self.grid_size[0]) / self.grid_step),
                                 int((self.grid_size[3] - self.grid_size[2]) / self.grid_step),
                                 int((self.grid_size[5] - self.grid_size[4]) / self.grid_step)),dtype=torch.float32,device=device)
        # Add coordinates
        grid_coordinates = torch.meshgrid(torch.arange(self.grid_size[0]+self.grid_step/2, self.grid_size[1], self.grid_step), torch.arange(self.grid_size[2]+self.grid_step/2, self.grid_size[3], self.grid_step), torch.arange(self.grid_size[4]+self.grid_step/2, self.grid_size[5], self.grid_step), indexing='ij')
        grid_coordinates = torch.stack(grid_coordinates, dim=0)
        self.grid[4:,:,:,:] = grid_coordinates

    def _generate_fixed_viewing_directions(self) -> torch.Tensor:
        phis = [math.pi / 3, 0, -math.pi]
        thetas = [k * math.pi / 3 for k in range(0, 6)]
        viewdirs = []

        for phi in phis:
            for theta in thetas:
                viewdirs.append(torch.Tensor([
                    math.cos(phi) * math.sin(theta),
                    math.cos(phi) * math.sin(theta),
                    math.sin(theta)
                ]))
        viewdirs = torch.stack(viewdirs, dim=0)
        return viewdirs
    
    def query_grid_one_batch(self, step, vision_field, renderer_rgb=None, batch_size=4096):
        if self.use_grid == False:
            return

        if self.grid == None:
            self.reset_grid()

        # We want a simple contraction of the coordinates to ensure we query the right grid cells
        vision_field.module.spatial_distortion = None
        aabb = vision_field.module.aabb.cpu()
        aabb_lengths = aabb[1] - aabb[0]

        i = self.grid_batch_i
    
        if i + batch_size > self.coordinates_to_render.shape[0]:
            batch_size = self.coordinates_to_render.shape[0] - i

        batch_coordinates = self.coordinates_to_render[i:i+batch_size]

        # we inverse the contraction
        ori = batch_coordinates
        ori = ori * aabb_lengths + aabb[0]

        start = torch.zeros_like(ori)
        end = torch.zeros_like(ori)

        if self.view_dirs is not None:
            rgbs = []
            densitys = []    
            oris = []
            dirs = []
            starts = []
            ends = []
            for j in range(len(self.view_dirs)):
                dir = self.view_dirs[j].expand(batch_size, -1)
                oris.append(ori)
                dirs.append(dir)
                starts.append(start)
                ends.append(end)
            f = Frustums(torch.cat(oris, dim=0),torch.cat(dirs, dim=0),torch.cat(starts, dim=0),torch.cat(ends, dim=0),None)
            camera_indices = torch.zeros((batch_size*len(self.view_dirs),1), dtype=torch.int)
            ray = RaySamples(f,camera_indices=camera_indices)

            ray = ray.to(self.device)

            field_outputs = vision_field.forward(ray)

            rgb = field_outputs[FieldHeadNames.RGB]
            density = field_outputs[FieldHeadNames.DENSITY]

            if renderer_rgb is not None:
                # volume rendering for rgb
                rgb = rgb.unsqueeze(1)
                density = density.unsqueeze(1)
                # weights = ray.get_weights(density)
                weights = torch.ones_like(density)
                rgb = renderer_rgb(rgb=rgb, weights=weights)

            for j in range(len(self.view_dirs)):
                rgbs.append(rgb[j*batch_size:(j+1)*batch_size])
                densitys.append(density[j*batch_size:(j+1)*batch_size])

            rgb = torch.mean(torch.stack(rgbs, dim=0), dim=0)
            density = torch.mean(torch.stack(densitys, dim=0), dim=0)

        else:
            dir = torch.tensor([1,0,0]).expand(batch_size, -1)
            f = Frustums(ori,dir,start,end,None)
            camera_indices = torch.zeros((batch_size,1), dtype=torch.int)
            ray = RaySamples(f,camera_indices=camera_indices)

            ray = ray.to(self.device)

            field_outputs = vision_field.forward(ray)

            rgb = field_outputs[FieldHeadNames.RGB]
            density = field_outputs[FieldHeadNames.DENSITY]

        batch_coordinates = batch_coordinates.to(self.grid.device)
        rgb = rgb.to(self.grid.device)
        density = density.to(self.grid.device)
            
        xs = ((batch_coordinates[:, 0] - self.grid_size[0]) / self.grid_step).int()
        ys = ((batch_coordinates[:, 1] - self.grid_size[2]) / self.grid_step).int()
        zs = ((batch_coordinates[:, 2] - self.grid_size[4]) / self.grid_step).int()

        mask = (xs >= 0) & (xs < self.grid.shape[1]) & (ys >= 0) & (ys < self.grid.shape[2]) & (zs >= 0) & (zs < self.grid.shape[3])

        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask]

        alpha = torch.clip(1 - torch.exp(-self._delta * density), 0, 1)
        if renderer_rgb is not None:
            color = rgb
        else:
            color = torch.sigmoid(rgb)

        color = color[mask]#.to(self.grid.device)
        alpha = alpha[mask]#.to(self.grid.device)

        self.grid = self.grid.detach()

        self.grid[0, xs, ys, zs] = color[:, 0].float().squeeze()
        self.grid[1, xs, ys, zs] = color[:, 1].float().squeeze()
        self.grid[2, xs, ys, zs] = color[:, 2].float().squeeze()
        self.grid[3, xs, ys, zs] = alpha.float().squeeze()

        self.grid_batch_i += batch_size
        if self.grid_batch_i >= self.coordinates_to_render.shape[0]:
            self.grid_batch_i = 0

        # we re-set the contraction 
        vision_field.module.spatial_distortion = self.spatial_distortion

    def query_grid(self, vision_field,batch_size=4096):
        print('WARNING: should be use to query the whole grid only in a torch no grad context')
        self.reset_grid()
        grid_size = self.grid_size
        grid_step = self.grid_step

        grid_coordinates = torch.meshgrid(torch.arange(grid_size[0]+grid_step/2, grid_size[1], grid_step), torch.arange(grid_size[2]+grid_step/2, grid_size[3], grid_step), torch.arange(grid_size[4]+grid_step/2, grid_size[5], grid_step), indexing='ij')
        grid_coordinates = torch.stack(grid_coordinates, dim=-1)
        coordinates_to_render = grid_coordinates.view(-1, 3)

        vision_field.module.spatial_distortion = None
        aabb = vision_field.module.aabb.cpu()
        aabb_lengths = aabb[1] - aabb[0]

        with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    MofNCompleteColumn(),
                    transient=True,
        ) as progress:
            task = progress.add_task("[green]Updating grid...", total=coordinates_to_render.shape[0])
            for i in range(0, coordinates_to_render.shape[0], batch_size):
                if i + batch_size > coordinates_to_render.shape[0]:
                    batch_size = coordinates_to_render.shape[0] - i

                batch_coordinates = coordinates_to_render[i:i+batch_size]
                
                ori = batch_coordinates
                ori = ori * aabb_lengths + aabb[0]

                start = torch.zeros_like(ori)
                end = torch.zeros_like(ori)

                if self.view_dirs is not None:
                    rgbs = []
                    densitys = []    
                    for j in range(len(self.view_dirs)):
                        dir = self.view_dirs[j].expand(batch_size, -1)
                        # dir = dir.to(self.device)
                        f = Frustums(ori,dir,start,end,None)
                        camera_indices = torch.zeros((batch_size,1), dtype=torch.int)
                        ray = RaySamples(f,camera_indices=camera_indices)

                        ray = ray.to(self.device)

                        field_outputs = vision_field.forward(ray)
                        rgb = field_outputs[FieldHeadNames.RGB]
                        density = field_outputs[FieldHeadNames.DENSITY]

                        rgbs.append(rgb.cpu())
                        densitys.append(density.cpu())

                    rgb = torch.mean(torch.stack(rgbs, dim=0), dim=0)
                    density = torch.mean(torch.stack(densitys, dim=0), dim=0)
                else:
                    dir = torch.tensor([1,0,0]).expand(batch_size, -1)
                    f = Frustums(ori,dir,start,end,None)
                    camera_indices = torch.zeros((batch_size,1), dtype=torch.int)
                    ray = RaySamples(f,camera_indices=camera_indices)

                    ray = ray.to(self.device)

                    field_outputs = vision_field.forward(ray)

                    rgb = field_outputs[FieldHeadNames.RGB]
                    density = field_outputs[FieldHeadNames.DENSITY]

                self.update_grid(batch_coordinates, rgb, density)
                progress.update(task, advance=batch_size)
        
        vision_field.module.spatial_distortion = self.spatial_distortion


    def update_grid(self, rays_xyz, rgb, density):
        if self.use_grid == False:
            return
            # pass #for debugging and visualization

        # when grid is completed with batch points
        # if self.spatial_distortion is not None:
        #     rays_xyz = self.spatial_distortion(rays_xyz)
        #     rays_xyz = (rays_xyz + 2.0) / 4.0
        # else:
        #     rays_xyz = SceneBox.get_normalized_positions(rays_xyz, self.scene_box.aabb)    
    
        rays_xyz = rays_xyz.reshape(-1, 3).to(self.grid.device)
        rgb = rgb.reshape(-1, 3).to(self.grid.device)
        density = density.reshape(-1, 1).to(self.grid.device)

        #apply sigmoid
        alpha = torch.clip(1 - torch.exp(-self._delta * density), 0, 1)
        color = torch.sigmoid(rgb)

        
        xs = ((rays_xyz[:, 0] - self.grid_size[0]) / self.grid_step).int()
        ys = ((rays_xyz[:, 1] - self.grid_size[2]) / self.grid_step).int()
        zs = ((rays_xyz[:, 2] - self.grid_size[4]) / self.grid_step).int()
        
        mask = (xs >= 0) & (xs < self.grid.shape[1]) & (ys >= 0) & (ys < self.grid.shape[2]) & (zs >= 0) & (zs < self.grid.shape[3])

        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask]
        color = color[mask]#.to(self.grid.device)
        alpha = alpha[mask]#.to(self.grid.device)

        self.grid[0, xs, ys, zs] = color[:, 0].float().squeeze()
        self.grid[1, xs, ys, zs] = color[:, 1].float().squeeze()
        self.grid[2, xs, ys, zs] = color[:, 2].float().squeeze()
        self.grid[3, xs, ys, zs] = alpha.float().squeeze()


        # #with actualization factor
        # actualization_factor = 0.1

        # self.grid[0, xs, ys, zs] = (1.0 - actualization_factor) * self.grid[0, xs, ys, zs] + actualization_factor * color[:, 0].float().squeeze()
        # self.grid[1, xs, ys, zs] = (1.0 - actualization_factor) * self.grid[1, xs, ys, zs] + actualization_factor * color[:, 1].float().squeeze()
        # self.grid[2, xs, ys, zs] = (1.0 - actualization_factor) * self.grid[2, xs, ys, zs] + actualization_factor * color[:, 2].float().squeeze()
        # self.grid[3, xs, ys, zs] = (1.0 - actualization_factor) * self.grid[3, xs, ys, zs] + actualization_factor * alpha.float().squeeze()


    def get_outputs(self, batch_audio):
        
        time_query = batch_audio['time_query'].to(self.device)
        time_query = time_query.float()/float(self.max_len - 1.0)
        time_query = time_query.unsqueeze(-1) 

        mic_pose = batch_audio['mic_pose'].to(self.device)
        source_pose = batch_audio['source_pose'].to(self.device)
        rot = batch_audio['rot'].to(self.device)
        
        mic_pose = SceneBox.get_normalized_positions(mic_pose, self.scene_box.aabb.cuda())
        source_pose = SceneBox.get_normalized_positions(source_pose, self.scene_box.aabb.cuda())
        selector = ((mic_pose > 0.0) & (mic_pose < 1.0)).all(dim=-1)
        mic_pose = mic_pose * selector[..., None]
        selector = ((source_pose > 0.0) & (source_pose < 1.0)).all(dim=-1)
        source_pose = source_pose * selector[..., None]

        mic_pose = self.position_encoding(mic_pose)
        source_pose = self.position_encoding(source_pose)
        rot = self.rot_encoding(rot)
        time_query = self.time_encoding(time_query)

        if self.use_grid:
            feat_grid = self.grid.unsqueeze(0)
            res_feat = self.resnet3d(feat_grid)
            feat_grid = res_feat 
            feat_grid = feat_grid.flatten()
            feat_grid = feat_grid.expand(time_query.shape[0], -1)

            h = torch.cat([feat_grid, time_query, mic_pose, source_pose, rot], dim=-1)
        else:
            h = torch.cat([mic_pose, source_pose, time_query, rot], dim=-1)

        field_outputs = self.field.forward(h.float())

        return field_outputs
    
    def get_metrics_dict(self, outputs, batch):
        with torch.no_grad():
            gt = batch['data']
            predicted = outputs.cpu().detach()

            # log pred w/ epsilon = 1e-3
            mag_prd = torch.clip(torch.exp(predicted) - 1e-3, 0.0, 10000.00)
            mag_gt = torch.clip(torch.exp(gt) - 1e-3, 0.0, 10000.00)

            # if mag pred 
            # mag_prd = predicted
            # mag_gt = gt

            return self.evaluator.get_stft_metrics(mag_prd, mag_gt)

    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):

        gt = batch['data'].to(self.device).float()
        predicted = outputs.float()

        # if mag pred
        # gt = torch.log(gt + 1e-3)
        # predicted = torch.log(predicted + 1e-3)
        
        loss = self.criterion(predicted, gt)
        if self.criterion_name == 'MSE':
            loss = {'audio_mse': loss * self.loss_factor}
        else:
            loss['audio_sc_loss'] = loss['audio_sc_loss'] * 1e-1 * self.loss_factor 
            loss['audio_mag_loss'] = loss['audio_mag_loss'] * 1.0 * self.loss_factor

        return loss 
    
    def set_eval_data(self, eval_source_pose, eval_mic_pose, eval_rot, eval_gt):
        #TMP for the viewer
        self.eval_source_pose = eval_source_pose
        self.eval_mic_pose = eval_mic_pose
        self.eval_rot = eval_rot
        self.eval_gt = eval_gt


    def get_outputs_for_camera(self, camera, obb_box, batch_audio=None):
        if camera is not None:
            # viewer is x: front, y: left, z: up
            # audio is x: front, y: up, z: left
            c2w_camera_ = camera.camera_to_worlds[0,:,:].cpu().numpy()# 0 because shape is [1, 3, 4]
            c2w_camera = np.eye(4)
            c2w_camera[:3, :3] = c2w_camera_[:3, :3]

            # convert to c2W to audio coordinate system using matrix multiplication
            transform_axis = np.array([[1, 0, 0, 0], 
                                       [0, 0, 1, 0], 
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])
            c2w_audio = np.matmul(transform_axis, c2w_camera)
            camera_pos = c2w_audio[:3, 3] 

            # source_pos = np.array(list(self.source_coordinates_gui.value))
            source_pos = np.eye(4)
            source_cam2world = np.matmul(transform_axis, source_pos)
            source_pos = source_cam2world[:3, 3]
            
            if self.dataset == 'RAF': 
                # Source Rotation
                rot = R.from_matrix(source_cam2world[:3, :3]).as_euler('yxz', degrees=True)
            else: 
                # Sound Spaces
                rot = R.from_matrix(c2w_camera[:3, :3]).as_euler('zyx', degrees=True)

            rot = np.round(rot[0], decimals=0)
            rad_rot = np.deg2rad(rot)
            rot = np.array([np.cos(rad_rot), 0, np.sin(rad_rot)])
            rot = (rot + 1.0) / 2.0

            mic_pose = torch.from_numpy(camera_pos).to(self.device)
            source_pose = torch.from_numpy(source_pos).to(self.device)
            rot = torch.from_numpy(rot).to(self.device)
            time_query = torch.arange(0, self.max_len, 1, device=self.device)

        else: #for evaluation
            time_query = torch.arange(0, self.max_len, 1, device=self.device)
            mic_pose = batch_audio['mic_pose'].to(self.device)
            source_pose = batch_audio['source_pose'].to(self.device)
            rot = batch_audio['rot'].to(self.device)
            self.set_eval_data(mic_pose, source_pose, rot, batch_audio['data'])

        with torch.no_grad():
            time_query = time_query.unsqueeze(-1).float()/float(self.max_len - 1.0)

            mic_pose = SceneBox.get_normalized_positions(mic_pose, self.scene_box.aabb.cuda())
            source_pose = SceneBox.get_normalized_positions(source_pose, self.scene_box.aabb.cuda())
            selector = ((mic_pose > 0.0) & (mic_pose < 1.0)).all(dim=-1)
            mic_pose = mic_pose * selector[..., None]
            selector = ((source_pose > 0.0) & (source_pose < 1.0)).all(dim=-1)
            source_pose = source_pose * selector[..., None]

            mic_pose = mic_pose.unsqueeze(0)
            source_pose = source_pose.unsqueeze(0)
            rot = rot.unsqueeze(0)

            mic_pose = self.position_encoding(mic_pose)
            source_pose = self.position_encoding(source_pose)
            rot = self.rot_encoding(rot)
            time_query = self.time_encoding(time_query)

            mic_pose = mic_pose.expand(time_query.shape[0], -1)
            source_pose = source_pose.expand(time_query.shape[0], -1)
            rot = rot.expand(time_query.shape[0], -1)

            if self.use_grid:
                
                feat_grid = self.grid.unsqueeze(0)
                feat_grid = feat_grid.to(self.device)
                res_feat = self.resnet3d(feat_grid)
                feat_grid = res_feat[-1]
                feat_grid = feat_grid.flatten()
                
                feat_grid = feat_grid.expand(time_query.shape[0], -1)

                h = torch.cat([feat_grid, time_query, mic_pose, source_pose, rot], dim=-1)
            else:
                h = torch.cat([mic_pose, source_pose, time_query, rot], dim=-1)

            field_outputs = self.field.forward(h.float())

            stft = {}
            for ch in range(field_outputs.shape[1]):
                v = field_outputs[:,ch,:].transpose(0,1).unsqueeze(-1).cpu()
                v = torch.flip(v, [0])
                # if mag pred
                # v = torch.log(v + 1e-3)
                stft["stft_ch_"+str(ch)] = v

            if camera is None:
                gt = self.eval_gt.to(self.device)
                for ch in range(gt.shape[0]):
                    v = gt[ch,:,:].unsqueeze(-1).cpu()
                    #flip the first dimension
                    v = torch.flip(v, [0])
                    # if mag pred
                    # v = torch.log(v + 1e-3)
                    stft["gt_ch_"+str(ch)] = v

                for ch in range(gt.shape[0]):
                    im = torch.cat([stft["stft_ch_"+str(ch)], stft["gt_ch_"+str(ch)]], dim=1)
                    stft["comparison_ch_"+str(ch)] = im

                if self.use_grid:
                    grid_colors = self.grid[0:3,:,:,:].cpu().detach().numpy()
                    grid_density = self.grid[3,:,:,:].cpu().detach().numpy()
                    grid_mean_over_z = np.mean(grid_colors, axis=3)
                    grid_density_mean_over_z = np.mean(grid_density, axis=2)
                    
                    stft["grid"] = torch.from_numpy(grid_mean_over_z).permute(1,2,0).to(self.device)
                    stft["grid_density"] = torch.from_numpy(grid_density_mean_over_z).unsqueeze(-1).to(self.device)

            if batch_audio is not None: #for metrics computation
                stft["raw_output"] = field_outputs

        return stft

    def get_param_groups(self): 
        # parameters
        param_groups = {}
        if self.use_grid:
            param_groups["audio_fields"] = list(self.field.parameters()) + list(self.rot_encoding.parameters()) + list(self.position_encoding.parameters()) + list(self.time_encoding.parameters()) + list(self.resnet3d.parameters())
        else:
            param_groups["audio_fields"] = list(self.field.parameters()) + list(self.rot_encoding.parameters()) + list(self.position_encoding.parameters()) + list(self.time_encoding.parameters())
        return param_groups
    
    def get_image_metrics_and_images(self, outputs, batch):
        with torch.no_grad():

            #outputs is T, C, F
            stft = outputs["raw_output"].permute(1,2,0).cpu() #shape [C, F, T]
            data = batch['data']

            mag_prd = torch.clip(torch.exp(stft) - 1e-3, 0.0, 10000.00)
            mag_gt = torch.clip(torch.exp(data) - 1e-3, 0.0, 10000.00)

            # load GT waveform 
            wav_gt = batch['waveform'].cpu().numpy()

            # istft using torch GriffinLim
            wav_istft_gt = self.istft_transform(mag_gt.cuda()).cpu().numpy()
            wav_istft_prd = self.istft_transform(mag_prd.cuda()).cpu().numpy()

            mag_prd = mag_prd.cpu().numpy()
            mag_gt = mag_gt.cpu().numpy()

            metrics_dict = self.evaluator.get_full_metrics(mag_prd, mag_gt, wav_gt, wav_istft_prd, wav_istft_gt, 
                                                stft.cpu().detach().numpy(), data.cpu().detach().numpy()) # log prd
                                                #   log_prd, log_gt) # mag prd
            
            images_dict = {}

            # retrieve min/max for normalization
            ids = []
            min_gt = np.inf
            max_gt = -np.inf
            for key in outputs.keys():
                if "gt" in key:
                    v = outputs[key]
                    ids.append(key.replace("gt_ch_",""))
                    min_gt = min(min_gt, v.min())
                    max_gt = max(max_gt, v.max())


            for id in ids:
                v = outputs["stft_ch_"+id]
                # #normalize using min and max of gt
                v = (v - min_gt) / (max_gt - min_gt)
                # #apply colormap
                v = cm.viridis(v.cpu().detach().numpy().squeeze())[...,:3]

                gt = outputs["gt_ch_"+id]
                gt = (gt - min_gt) / (max_gt - min_gt)
                gt = cm.viridis(gt.cpu().detach().numpy().squeeze())[...,:3]

                diff = outputs["stft_ch_"+id] - outputs["gt_ch_"+id]
                diff = (diff - min_gt) / (max_gt - min_gt)
                diff = cm.viridis(diff.cpu().detach().numpy().squeeze())[...,:3]

                im = np.concatenate([v, gt], axis=1)
                images_dict["comparison_ch_"+id] = torch.from_numpy(im)
                # images_dict["stft_ch_"+id] = torch.from_numpy(v)
                # images_dict["gt_ch_"+id] = torch.from_numpy(gt)

            if self.use_grid:
                images_dict["grid"] = outputs["grid"]
                im_grid_density = outputs["grid_density"]
                im_grid_density = (im_grid_density - im_grid_density.min()) / (im_grid_density.max() - im_grid_density.min())
                im_grid_density = cm.viridis(im_grid_density.cpu().detach().numpy().squeeze())[...,:3]
                
                images_dict["grid_density"] = torch.from_numpy(im_grid_density)

        return metrics_dict, images_dict