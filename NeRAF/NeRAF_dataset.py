"""
SoundSpaces Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import os
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from NeRAF.NeRAF_dataparser import SoundSpacesDataparserOutputs, RAFDataparserOutputs

import librosa
import scipy.io.wavfile as wavfile

""""""""""""" RAF """""""""""""
class RAFDataset(Dataset):
    """Dataset that returns audios.

    Args:
        dataparser_outputs: description of where and how to read input audios.
    """

    exclude_batch_keys_from_device: List[str] = ["audios"]

    def __init__(self, dataparser_outputs: RAFDataparserOutputs, 
                 mode: Literal["train", "eval", "inference"] = "train",
                 max_len: int = 100, 
                 max_len_time: float = 0.32,
                 wav_path: Path = None,
                 fs: int = 48000,
                 hop_len: int = 256, 
                 mean = None,
                 std = None):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.audios_filenames=self._dataparser_outputs.audios_filenames
        self.microphone_poses=self._dataparser_outputs.microphone_poses
        self.source_poses=self._dataparser_outputs.source_poses
        self.source_rotations=self._dataparser_outputs.source_rotations
        self.scene_box=self._dataparser_outputs.scene_box

        self.wav_path = wav_path

        self.fs = fs

        if self.fs == 48000:
            self.n_fft = 1024
            self.win_length = 512
            self.hop_len = 256
        elif self.fs == 16000:
            self.n_fft = 512
            self.win_length = 256
            self.hop_len = 128
        else: 
            raise ValueError('Sample rate not supported')
        
        self.transform_stft_torch = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_len, power=None)

        self.max_len = max_len
        self.max_len_time = max_len_time

        self.mode = mode 

        self.mean = mean
        self.std = std

    def __len__(self):
        if self.mode == "train":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == "eval":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == "inference":
            return len(self._dataparser_outputs.audios_filenames)
        return len(self._dataparser_outputs.audios_filenames)
    
    def get_id_tmp(self, idx):
        return idx//self.max_len, idx%self.max_len
    
    def get_data(self, audio_idx: int):
        """Returns the of shape STFT slices of size(C, F), microphone pose, source pose, and microphone rotation.
        """
        
        stft_id, stft_tp = self.get_id_tmp(audio_idx)
        audio_filename = self._dataparser_outputs.audios_filenames[stft_id]

        audio_folder = os.path.join(self.wav_path, audio_filename)
        wav_file = os.path.join(audio_folder, 'rir.wav')
        data, sr = librosa.load(wav_file, sr=None)
        if sr != 48000:
            raise ValueError('Loaded sample rate should be 48kHz')
        
        if self.fs != sr:
            init_fs = sr
            if data.shape[1]<int(init_fs*0.1):
                padded_wav = librosa.util.fix_length(data, int(init_fs*0.1))
                resampled_wav= librosa.resample(padded_wav, orig_sr=init_fs, target_sr=self.fs)
            else:
                resampled_wav= librosa.resample(data, orig_sr=init_fs, target_sr=self.fs)
            data = resampled_wav
   
        data = data[:self.max_len_time]

        stft = self.transform_stft_torch(torch.tensor(data))
        stft = stft.unsqueeze(0)

        if stft_tp < stft.shape[2]:
            stft = torch.log(torch.abs(stft[:,:,stft_tp]) + 1e-3)
        else:
            min_value = torch.min(stft)
            stft = torch.ones(stft.shape[0], stft.shape[1]) * min_value
            stft = torch.log(torch.abs(stft) + 1e-3)


        # get poses
        microphone_pose = self._dataparser_outputs.microphone_poses[stft_id]
        source_pose = self._dataparser_outputs.source_poses[stft_id]
        source_rotations = self._dataparser_outputs.source_rotations[stft_id]

        data = {"audio_idx": stft_id, "data": stft, "time_query": stft_tp,
                'rot': source_rotations, 'mic_pose': microphone_pose, 'source_pose': source_pose}

        return data

        
    def get_data_eval(self, audio_idx: int):
        """Returns the STFT of shape (C, F, T), microphone pose, source pose, and microphone rotation.

        Args:
            audio_idx: The audio index in the dataset.
        """
        audio_filename = self._dataparser_outputs.audios_filenames[audio_idx]

        audio_folder = os.path.join(self.wav_path, audio_filename)
        wav_file = os.path.join(audio_folder, 'rir.wav')
        data, sr = librosa.load(wav_file, sr=None)
        if sr != 48000:
            raise ValueError('Loaded sample rate should be 48kHz')
        
        if self.fs != sr:
            init_fs = sr
            if data.shape[1]<int(init_fs*0.1):
                padded_wav = librosa.util.fix_length(data, int(init_fs*0.1))
                resampled_wav= librosa.resample(padded_wav, orig_sr=init_fs, target_sr=self.fs)
            else:
                resampled_wav= librosa.resample(data, orig_sr=init_fs, target_sr=self.fs)
            data = resampled_wav

        data = data[:self.max_len_time]
        
        # create stft
        stft = self.transform_stft_torch(torch.tensor(data))
        stft = stft.unsqueeze(0)

        if stft.shape[2] > self.max_len:
            stft = stft[:,:,:self.max_len]
        else:
            min_value = stft.min()
            stft = torch.nn.functionnal.pad(stft, ((0,0), (0,0), (0, self.max_len - stft.shape[2])), 'constant', constant_values=min_value)

        stft = torch.log(torch.abs(stft) + 1e-3)

        waveform = torch.from_numpy(data)
        waveform = waveform.unsqueeze(0)

        # get poses
        microphone_pose = self._dataparser_outputs.microphone_poses[audio_idx]
        source_pose = self._dataparser_outputs.source_poses[audio_idx]
        source_rotations = self._dataparser_outputs.source_rotations[audio_idx]

        data = {"audio_idx": audio_idx, "data": stft, 'waveform': waveform, 
                'rot': source_rotations, 'mic_pose': microphone_pose, 'source_pose': source_pose}

        return data
    
    def get_data_inference(self, audio_idx: int):
        # create zeros stft and waveform because we don't have GT, it's only for inferences
        stft = torch.zeros(1, (self.n_fft // 2)+1, self.max_len)
        waveform = torch.zeros(1, self.max_len_time)

        # get poses
        microphone_pose = self._dataparser_outputs.microphone_poses[audio_idx]
        source_pose = self._dataparser_outputs.source_poses[audio_idx]
        source_rotations = self._dataparser_outputs.source_rotations[audio_idx]

        data = {"audio_idx": audio_idx, "data": stft, 'waveform': waveform, 
                'rot': source_rotations, 'mic_pose': microphone_pose, 'source_pose': source_pose}
        
        return data


    def __getitem__(self, image_idx: int) -> Dict:
        if self.mode == "train":
            return self.get_data(image_idx)
        elif self.mode == "eval":
            return self.get_data(image_idx) # special case for evaluation on not full STFT
        elif self.mode == "inference":
            return self.get_data_inference(image_idx)
        return self.get_data_eval(image_idx)

    @property
    def audio_filenames(self) -> List[Path]:
        """
        Returns audio filenames for this dataset.
        """
        return self._dataparser_outputs.audio_filenames



""""""""""""" Sound Spaces """""""""""""
class SoundSpacesDataset(Dataset):
    """Dataset that returns audios.

    Args:
        dataparser_outputs: description of where and how to read input audios.
    """

    exclude_batch_keys_from_device: List[str] = ["audios"]

    def __init__(self, dataparser_outputs: SoundSpacesDataparserOutputs, 
                 mode: Literal["train", "eval", "inference"] = "train",
                 max_len: int = 100, 
                 mag_path: Path = None, 
                 wav_path: Path = None,
                 fs: int = 22050,
                 hop_len: int = 128, 
                 mean = None,
                 std = None,):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.audios_filenames=self._dataparser_outputs.audios_filenames
        self.microphone_poses=self._dataparser_outputs.microphone_poses
        self.source_poses=self._dataparser_outputs.source_poses
        self.microphone_rotations=self._dataparser_outputs.microphone_rotations
        self.scene_box=self._dataparser_outputs.scene_box

        self.mag_path = mag_path    
        self.wav_path = wav_path

        self.fs = fs
        self.hop_len = hop_len

        self.max_len = max_len  
        self.max_len_time = self.max_len * self.hop_len 

        self.mode = mode

        self.mean = mean
        self.std = std

    def __len__(self):
        if self.mode == "train":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == "eval":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == 'inference': 
            return len(self._dataparser_outputs.audios_filenames) 
        return len(self._dataparser_outputs.audios_filenames)
    
    def get_id_tmp(self, idx):
        return idx//self.max_len, idx%self.max_len
    
    def get_data(self, audio_idx: int):
        """Returns the of shape STFT slices of size(C, F), microphone pose, source pose, and microphone rotation.
        """
        
        stft_id, stft_tp = self.get_id_tmp(audio_idx)
        audio_filename = self._dataparser_outputs.audios_filenames[stft_id]

        # Load STFT
        stft = np.load(os.path.join(self.mag_path, audio_filename + '.npy'))
        stft = torch.from_numpy(stft)
        if stft_tp < stft.shape[2]:
            stft = torch.log(stft[:,:,stft_tp] + 1e-3)
        else:
            min_value = stft.min()
            stft = torch.ones(stft.shape[0], stft.shape[1]) * min_value
            stft = torch.log(stft + 1e-3)

        # Get poses from dataparser
        microphone_pose = self._dataparser_outputs.microphone_poses[stft_id]
        source_pose = self._dataparser_outputs.source_poses[stft_id]
        microphone_rotation = self._dataparser_outputs.microphone_rotations[stft_id]

        data = {"audio_idx": stft_id, "data": stft, "time_query": stft_tp,
                'rot': microphone_rotation, 'mic_pose': microphone_pose, 'source_pose': source_pose}
        return data


    def get_data_eval(self, audio_idx: int):
        """Returns the STFT of shape (C, F, T), microphone pose, source pose, and microphone rotation.

        Args:
            audio_idx: The audio index in the dataset.
        """
        audio_filename = self._dataparser_outputs.audios_filenames[audio_idx]

        if self.mode == "inference":
            # all zeros stft because we don't have GT
            stft = torch.zeros((2, 257, self.max_len))
            waveform = torch.zeros((2, int(self.max_len_time)))
        
        else:
            # Load STFT
            stft = np.load(os.path.join(self.mag_path, audio_filename + '.npy'))
            if stft.shape[2] > self.max_len:
                stft = stft[:,:,:self.max_len]
            else:
                min_value = stft.min()
                stft = np.pad(stft, ((0,0), (0,0), (0, self.max_len - stft.shape[2])), 'constant', constant_values=min_value)

            stft = np.log(stft + 1e-3)
            stft = torch.from_numpy(stft) 
            
            # Load GT waveform for metric computation
            loaded = wavfile.read(os.path.join(self.wav_path, audio_filename + '.wav'))
            waveform = np.clip(loaded[1], -1.0, 1.0).T

            if waveform.shape[1] == 0:
                waveform = np.zeros((2, int(self.fs*0.5)))

            if self.fs != 44100:
                init_fs = 44100
                if waveform.shape[1]<int(init_fs*0.1):
                    padded_wav = librosa.util.fix_length(data=waveform, size=int(init_fs*0.1))
                    resampled_wav= librosa.resample(padded_wav, orig_sr=init_fs, target_sr=self.fs)
                else:
                    resampled_wav= librosa.resample(waveform, orig_sr=init_fs, target_sr=self.fs)
                waveform = resampled_wav

            if waveform.shape[1] > self.max_len_time:
                waveform = waveform[:,:int(self.max_len_time)]
            else:
                waveform = np.pad(waveform, ((0,0), (0, self.max_len_time - waveform.shape[1])), 'constant', constant_values=0)

            waveform = torch.from_numpy(waveform)

        # Get poses from dataparser
        microphone_pose = self._dataparser_outputs.microphone_poses[audio_idx]
        source_pose = self._dataparser_outputs.source_poses[audio_idx]
        microphone_rotation = self._dataparser_outputs.microphone_rotations[audio_idx]

        data = {"audio_idx": audio_idx, "data": stft, 'waveform': waveform, 
                'rot': microphone_rotation, 'mic_pose': microphone_pose, 'source_pose': source_pose}

        return data

    def __getitem__(self, image_idx: int) -> Dict:
        if self.mode == "train":
            return self.get_data(image_idx)
        elif self.mode == "eval":
            return self.get_data(image_idx) # special case for evaluation on STFT slices
        return self.get_data_eval(image_idx)

    @property
    def audio_filenames(self) -> List[Path]:
        """
        Returns audio filenames for this dataset.
        """
        return self._dataparser_outputs.audio_filenames