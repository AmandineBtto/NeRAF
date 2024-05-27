"""
Audio Datamanager for NeRAF
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

from nerfstudio.cameras.cameras import Cameras
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)


import os
import json
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa 

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn


@dataclass
class NeRAFDataManagerConfig(DataManagerConfig):
    """NeRAF DataManager Config
    """

    _target: Type = field(default_factory=lambda: NeRAFDataManager)
    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096
    max_len: int = 156
    hop_len: int = 128
    fs: int = 44100


class NeRAFDataManager(DataManager):
    """NeRAF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NeRAFDataManagerConfig

    def __init__(
        self,
        config: NeRAFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.config = config

        self.max_len = config.max_len
        self.hop_len = config.hop_len
        self.fs = config.fs

        self.train_pixel_sampler = None # not used for audio
        
        path_query = os.path.join(self.config.data, 'metadata/points.txt')
        
        with open(path_query, "r") as f:
                lines = f.readlines()
        coords = [x.replace("\n", "").split("\t") for x in lines] # contains xyz 
        positions = dict() 
        for row in coords:
            readout = [float(xyz) for xyz in row[1:]]
            positions[row[0]] = [readout[0], readout[2], -readout[1]] # up is second 

        self.positions = positions

        with open(os.path.join(self.config.data, 'metadata_AudioNeRF/split.json'), 'r') as f: 
            split_dict = json.load(f)
        
        audio_train_files = split_dict['train']
        self.audio_train_files = audio_train_files
        audio_test_files = split_dict['test']
        N_test = len(audio_test_files)
        self.audio_test_files = audio_test_files[:N_test]
        print('Number of train files', len(self.audio_train_files))
        print('Number of test files', len(self.audio_test_files))    
        
        if self.fs == 44100:
            self.mag_path = os.path.join(self.config.data, 'binaural_magnitudes')
        else:
            self.mag_path = os.path.join(self.config.data, 'binaural_magnitudes_sr22050')
        self.wav_path = os.path.join(self.config.data, 'binaural_rirs')


        print("Loading Audio training dataset...")
        self.train_dataset = self._load_stft(self.audio_train_files)
        print("Loading Audio test dataset...")
        self.test_dataset = self._load_stft(self.audio_test_files)
        self.test_waveform = self._load_waveform(self.audio_test_files)
        print("Data loaded")

        self.num_sample_train = self.train_dataset[0].shape[0] * self.train_dataset[0].shape[3]
        self.train_batchs = self.create_batch_indices(self.num_sample_train, self.config.train_num_rays_per_batch)
        self.num_sample_eval = self.test_dataset[0].shape[0] * self.test_dataset[0].shape[3]
        self.eval_batchs = self.create_batch_indices(self.num_sample_eval, self.config.eval_num_rays_per_batch)

        self.train_count = 0
        self.eval_count = 0
        self.eval_image_count = 0

    def get_id_tmp(self, idx):
        return idx//self.train_dataset[0].shape[3], idx%self.train_dataset[0].shape[3]

    def create_batch_indices(self, num_samples: int, num_per_batch) -> torch.Tensor:
        #custom batchification
        permuted_indices = self.generate_permuted_indices(num_samples)
        batchs = []
        for i in range(0, num_samples, num_per_batch):
            batchs.append(permuted_indices[i:i+num_per_batch])
        #delete last if not full
        if len(batchs[-1]) < num_per_batch:
            batchs.pop()
        return batchs

    def generate_permuted_indices(self, num_samples: int) -> torch.Tensor:
        """Generates a random permutation of indices."""
        return torch.randperm(num_samples)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        
        batch_indices = self.train_batchs[self.train_count]
        batch_id_tp = [self.get_id_tmp(idx) for idx in batch_indices] 
        batch_id = [x[0] for x in batch_id_tp]
        batch_tp = [x[1] for x in batch_id_tp]

        data = self.train_dataset[0][batch_id, ..., batch_tp]
        rot = self.train_dataset[1][batch_id]
        mic_pose = self.train_dataset[2][batch_id]
        source_pose = self.train_dataset[3][batch_id]


        batch = {'rot': rot, 'mic_pose': mic_pose, 'source_pose': source_pose, 'data': data, 'time_query': batch_tp}

        self.train_count += 1
        if self.train_count >= len(self.train_batchs):
            self.train_count = 0
            self.train_batchs = self.create_batch_indices(self.num_sample_train, self.config.train_num_rays_per_batch)

        return None, batch
    

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        batch_indices = self.eval_batchs[self.eval_count]
        batch_id_tp = [self.get_id_tmp(idx) for idx in batch_indices]
        batch_id = [x[0] for x in batch_id_tp]
        batch_tp = [x[1] for x in batch_id_tp]

        data = self.test_dataset[0][batch_id, ..., batch_tp]
        rot = self.test_dataset[1][batch_id]
        mic_pose = self.test_dataset[2][batch_id]
        source_pose = self.test_dataset[3][batch_id]

        batch = {'rot': rot, 'mic_pose': mic_pose, 'source_pose': source_pose, 'data': data, 'time_query': batch_tp}

        self.eval_count += 1
        if self.eval_count >= len(self.eval_batchs):
            self.eval_count = 0
            self.eval_batchs = self.create_batch_indices(self.num_sample_eval, self.config.eval_num_rays_per_batch)
    
        return None, batch
    
    def next_eval_image(self, step):
        i = self.eval_image_count
        data = self.test_dataset[0][i]
        rot = self.test_dataset[1][i]
        mic_pose = self.test_dataset[2][i]
        source_pose = self.test_dataset[3][i]
        waveform = self.test_waveform[i]
        batch = {'rot': rot, 'mic_pose': mic_pose, 'source_pose': source_pose, 'data': data, 'waveform': waveform}
        
        self.eval_image_count += 1
        if self.eval_image_count >= len(self.test_dataset[0]):
            self.eval_image_count = 0

        return None, batch

    
    def _process_poses(self, f):
        rot, r_s = f.split('/')
        rot = int(rot)
        mic_pose = self.positions[r_s.split('_')[0]][:3]
        source_pose = self.positions[r_s.split('_')[1]][:3]
        mic_pose = np.expand_dims(np.array(mic_pose), axis=0)
        source_pose = np.expand_dims(np.array(source_pose), axis=0)
        rot = np.expand_dims(np.array([rot]), axis=0)
        return {'rot': rot, 'mic_pose': mic_pose, 'source_pose': source_pose}
    
    def _process_stft(self, f):
        data = np.load(os.path.join(self.mag_path, f + '.npy'))
        if data.shape[2] > self.max_len:
            data = data[:,:,:self.max_len]
        else:
            min_value = data.min()
            data = np.pad(data, ((0,0), (0,0), (0, self.max_len - data.shape[2])), 'constant', constant_values=min_value)

        data = np.log(data + 1e-3)
        data = np.expand_dims(data, axis=0)

        # get mics and sources poses
        dict_audio = self._process_poses(f)

        dict_audio['data'] = data

        return dict_audio

    def _load_stft(self, files):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Loading dataset...", total=len(files))
            all_dict = []
            for f in files:
                dic = self._process_stft(f)
                all_dict.append(dic)

                progress.advance(task)
            all_data = np.concatenate([d['data'] for d in all_dict], axis=0)
            all_rot = np.concatenate([d['rot'] for d in all_dict], axis=0)
            all_mic_pose = np.concatenate([d['mic_pose'] for d in all_dict], axis=0)
            all_source_pose = np.concatenate([d['source_pose'] for d in all_dict], axis=0)

        return all_data, all_rot, all_mic_pose, all_source_pose
    
    def _process_wav(self, f):
        loaded = wavfile.read(os.path.join(self.wav_path, f + '.wav'))
        data = np.clip(loaded[1], -1.0, 1.0).T

        if self.fs != 44100:
            init_fs = 44100
            if data.shape[1]<int(init_fs*0.1):
                padded_wav = librosa.util.fix_length(data, int(init_fs*0.1))
                resampled_wav= librosa.resample(padded_wav, orig_sr=init_fs, target_sr=self.fs)
            else:
                resampled_wav= librosa.resample(data, orig_sr=init_fs, target_sr=self.fs)
            data = resampled_wav

        max_len_time = self.max_len * self.hop_len 
        if data.shape[1] > max_len_time:
            data = data[:,:int(max_len_time)]
        else:
            data = np.pad(data, ((0,0), (0, max_len_time - data.shape[1])), 'constant', constant_values=0)

        data = np.expand_dims(data, axis=0)
        return {'data': data}
    
    def _load_waveform(self, files): 
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Loading eval waveform...", total=len(files))
            all_dict = []
            for f in files:
                dic = self._process_wav(f)
                all_dict.append(dic)
                progress.advance(task)
            all_data = np.concatenate([d['data'] for d in all_dict], axis=0)

        return all_data
