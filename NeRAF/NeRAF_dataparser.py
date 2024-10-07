from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

import torch
import numpy as np
import pickle as pkl
import os
from jaxtyping import Float
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs

from scipy.spatial.transform import Rotation as T

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from nerfstudio.data.scene_box import SceneBox

""""""""""""" NeRAF """""""""""""
@dataclass
class NeRAFDataparserOutputs(DataparserOutputs):

    def __init__(self, audios_filenames, microphone_poses, source_poses, scene_box, microphone_rotations=None, source_rotations=None):
        #super().__init__()
        self.audios_filenames = audios_filenames    
        self.microphone_poses = microphone_poses
        self.microphone_rotations = microphone_rotations
        self.source_poses = source_poses
        self.source_rotations = source_rotations
        self.scene_box = scene_box

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)
    
@dataclass
class NeRAFDataParserConfig(DataParserConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: NeRAFDataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""


@dataclass
class NeRAFDataParser(DataParser):
    
    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> RAFDataparserOutputs:
        print("Abstract method, Not implemented")



    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> NeRAFDataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split, **kwargs)
        return dataparser_outputs
    
    @abstractmethod
    def _process_poses(self, files):
        print("Abstract method, Not implemented")


    @abstractmethod
    def _load_poses_eval(self, eval_data):
        print("Abstract method, Not implemented")


""""""""""""" RAF """""""""""""
@dataclass
class RAFDataparserOutputs(NeRAFDataparserOutputs):
    """Dataparser outputs for the which will be used by the DataManager
    for creating STFTTime and STFTGT objects."""

    def __init__(self, audios_filenames, microphone_poses, source_poses, source_rotations, scene_box):
        super().__init__(audios_filenames, microphone_poses, source_poses, scene_box, source_rotations=source_rotations)

@dataclass
class RAFDataParserConfig(NeRAFDataParserConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: RAFDataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""

@dataclass
class RAFDataParser(NeRAFDataParser):
    """A dataset.

    Args:
        config: datasetparser config containing all information needed to instantiate dataset

    Attributes:
        config: datasetparser config containing all information needed to instantiate dataset
    """

    config: RAFDataParserConfig

    def __init__(self, 
                config: RAFDataParserConfig):
        super().__init__(config)
        self.config = config

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> RAFDataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """

        if split == 'inference': # special case for inference
            # to launch inference:
            # AVN_RENDER_POSES=poses2render.npy ns-eval --load-config ./path2/config.yml 
            # --output-path ./video_folder/results.json 
            # --render-output-path ./video_folder
            path_eval_poses = os.environ['AVN_RENDER_POSES']
            print('Render poses in', path_eval_poses)
            poses = self._load_poses_eval(path_eval_poses)
            split_files = list(range(0, poses['mic_pose'].shape[0]))

        else:
            split_file = os.path.join(self.config.data, 'metadata/data-split.json')
            with open(split_file) as f:
                split_files = json.load(f)

            if split == 'train':
                split_files = split_files['train'][0]
            elif split == 'val':
                split_files = split_files['validation'][0]
            else:
                split = "test" 
                split_files = split_files['test'][0]

            poses = self._process_poses(split_files)

        # test: aabb based on poses['mic_pose']
        aabb = np.array([poses['mic_pose'].min(axis=0), poses['mic_pose'].max(axis=0)])
        # add 1m margin 
        aabb[0] -= 1
        aabb[1] += 1

        scene_box = SceneBox(
            aabb = torch.tensor(aabb, dtype=torch.float32)) # min max on each axis 
        
        dataparser_outputs = RAFDataparserOutputs(
            audios_filenames= split_files, #list(split_files.keys()),
            microphone_poses=torch.from_numpy(poses['mic_pose']),
            source_poses=torch.from_numpy(poses['source_pose']),
            source_rotations=torch.from_numpy(poses['rot']),    
            scene_box=scene_box, 
        )

        return dataparser_outputs


    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> RAFDataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split, **kwargs)
        return dataparser_outputs
    
    def _process_poses(self, files):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Loading Poses...", total=len(files))

            c=0
            for f in files:
                rx_file = os.path.join(self.config.data, 'data', f, 'rx_pos.txt')
                tx_file = os.path.join(self.config.data, 'data', f, 'tx_pos.txt')
                with open(rx_file, 'r') as f:
                    rx = f.readlines()
                    rx = [i.replace('\n','').split(',') for i in rx]
                    rx = [[float(j) for j in i] for i in rx]
                    rx = rx[0]
                with open(tx_file, 'r') as f:
                    tx = f.readlines()
                    tx = [i.replace('\n','').split(',') for i in tx]
                    tx = [[float(j) for j in i] for i in tx]
                    tx = tx[0]

                quat = tx[:4] #xyzW
                tx_pose = tx[4:]
                rx_pose = rx 

                r = T.from_quat(quat)
                view_dir = np.array([1, 0, 0])
                spk_viewdir = r.apply(view_dir)

                spk_rot = r.as_euler('yxz', degrees=True)
                spk_rot = np.round(spk_rot[0], decimals=0) # rotation around y axis only
                rad_spk_rot = np.deg2rad(spk_rot)
                spk_rot = np.array([np.cos(rad_spk_rot), 0, np.sin(rad_spk_rot)]) # to use for SHE
                spk_rot = (spk_rot + 1.0) / 2.0

                mic_pose = np.expand_dims(np.array(rx_pose), axis=0)
                source_pose = np.expand_dims(np.array(tx_pose), axis=0)
                rot = np.expand_dims(np.array(spk_rot, dtype=float), axis=0)


                if c == 0:
                    mic_poses = mic_pose
                    source_poses = source_pose
                    rots = rot
                else:
                    mic_poses = np.concatenate((mic_poses, mic_pose), axis=0)
                    source_poses = np.concatenate((source_poses, source_pose), axis=0)
                    rots = np.concatenate((rots, rot), axis=0)

                c+=1
                progress.advance(task)

        return {'rot': rots, 'mic_pose': mic_poses, 'source_pose': source_poses}

    def _load_poses_eval(self, eval_data):
        print('Loading poses for inference')

        data = np.load(eval_data, allow_pickle=True).item()
        mic_poses = data['mic_poses']

        source_poses = data['source_poses']
        rots = data['rots']

        rots = np.expand_dims(rots, axis=0)
        rots = np.repeat(rots, mic_poses.shape[0], axis=0)
        source_poses = np.expand_dims(source_poses, axis=0)
        source_poses = np.repeat(source_poses, mic_poses.shape[0], axis=0)
    
        return {'rot': rots, 'mic_pose': mic_poses, 'source_pose': source_poses}

    


""""""""""""" Sound Spaces """""""""""""
@dataclass
class SoundSpacesDataparserOutputs(NeRAFDataparserOutputs):
    """Dataparser outputs for the which will be used by the DataManager
    for creating STFT Time and STFT GT objects."""

    def __init__(self, audios_filenames, microphone_poses, source_poses, microphone_rotations, scene_box):
        super().__init__(audios_filenames, microphone_poses, source_poses, scene_box, microphone_rotations=microphone_rotations)

@dataclass
class SoundSpacesDataParserConfig(NeRAFDataParserConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: SoundSpacesDataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""

@dataclass
class SoundSpacesDataParser(NeRAFDataParser):
    """Sound Spaces DataParser.
    """

    config: SoundSpacesDataParserConfig

    def __init__(self, 
                config: SoundSpacesDataParserConfig):
        super().__init__(config)
        self.config = config

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> SoundSpacesDataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        path_query = os.path.join(self.config.data, 'metadata/points.txt')
        with open(path_query, "r") as f:
                lines = f.readlines()
        coords = [x.replace("\n", "").split("\t") for x in lines] # contains xyz 
        positions = dict() 
        for row in coords:
            readout = [float(xyz) for xyz in row[1:]]
            positions[row[0]] = [readout[0], readout[2], -readout[1]] # up is second axis
        self.positions = positions

        if split == 'inference': # special case for inference
            # to launch inference use:
            # AVN_RENDER_POSES=poses2render.npy ns-eval --load-config ./path2/config.yml 
            # --output-path ./video_folder/results.json 
            # --render-output-path ./video_folder
            path_eval_poses = os.environ['AVN_RENDER_POSES']
            print('Render poses in', path_eval_poses)
            eval_data = pkl.load(open(path_eval_poses, "rb"))
            eval_data = eval_data["scene_obs"]

            poses = self._load_poses_eval(eval_data)
            split_files = eval_data

        else:
            with open(os.path.join(self.config.data, 'metadata_AudioNeRF/split.json'), 'r') as f: 
                split_dict = json.load(f)
            
            if split == 'train':
                split_files = split_dict[split]
            else:
                split = "test" # No validation set in SoundSpaces
                split_files = split_dict[split]

            poses = self._process_poses(split_files)

        # Create audio AABB
        aabb = np.array([poses['mic_pose'].min(axis=0), poses['mic_pose'].max(axis=0)])
        # add a 1m margin 
        aabb[0] -= 1.0
        aabb[1] += 1.0

        scene_box = SceneBox(
            aabb = torch.tensor(aabb, dtype=torch.float32)) 
    
        dataparser_outputs = SoundSpacesDataparserOutputs(
            audios_filenames=split_files,
            microphone_poses=torch.from_numpy(poses['mic_pose']),
            source_poses=torch.from_numpy(poses['source_pose']),
            microphone_rotations=torch.from_numpy(poses['rot']),   
            scene_box=scene_box, 
        )

        return dataparser_outputs


    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> SoundSpacesDataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split, **kwargs)
        return dataparser_outputs
    
    def _process_poses(self, files):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Loading Poses...", total=len(files))

            c=0
            for f in files:
                rot, r_s = f.split('/')
                rot = int(rot)
                mic_pose = self.positions[r_s.split('_')[0]][:3]
                source_pose = self.positions[r_s.split('_')[1]][:3]
                mic_pose = np.expand_dims(np.array(mic_pose), axis=0)
                source_pose = np.expand_dims(np.array(source_pose), axis=0)
                # From the angle around up-axis, we want a direction cosine (necessary for SHE)
                rot = np.deg2rad(rot)
                rot = np.array([np.cos(rot), 0, np.sin(rot)]) 
                rot = (rot + 1.0) / 2.0 # normalize (should be btw 0 1 for TCNN)
                rot = np.expand_dims(rot, axis=0)

                if c == 0:
                    mic_poses = mic_pose
                    source_poses = source_pose
                    rots = rot
                else:
                    mic_poses = np.concatenate((mic_poses, mic_pose), axis=0)
                    source_poses = np.concatenate((source_poses, source_pose), axis=0)
                    rots = np.concatenate((rots, rot), axis=0)

                c+=1
                progress.advance(task)

        return {'rot': rots, 'mic_pose': mic_poses, 'source_pose': source_poses}

    def _load_poses_eval(self, eval_data):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Loading Eval Poses...", total=len(eval_data))
            c = 0
            for v in eval_data:
                pose = v["pose"]
                quat = v["quat"]
                mic_pose = pose[:3]
                # Get rotation in degree around up-axis
                quat = T.from_quat(quat) 
                quat = quat.as_euler("yzx", degrees=True)
                mic_rot = quat[0]
                if mic_rot < 0: # necessary because offset btw Habitat and SoundSpaces audio 
                    mic_rot = 360 + mic_rot
                mic_rot = mic_rot % 360
                # Get direction cosine
                rot = np.deg2rad(mic_rot)
                rot = np.array([np.cos(rot), 0, np.sin(rot)]) 
                rot = (rot + 1.0) / 2.0
                rot = np.array([rot])  

                source_pose = v["source"]
                source_pose = source_pose[:3]
                # Set mic height to source height because train done with a fixed height
                mic_pose[1] = source_pose[1]
                
                mic_pose = np.expand_dims(np.array(mic_pose), axis=0)
                source_pose = np.expand_dims(np.array(source_pose), axis=0)

                if c == 0:
                    mic_poses = mic_pose
                    source_poses = source_pose
                    rots = rot
                else:
                    mic_poses = np.concatenate((mic_poses, mic_pose), axis=0)
                    source_poses = np.concatenate((source_poses, source_pose), axis=0)
                    rots = np.concatenate((rots, rot), axis=0)

                c+=1
                progress.advance(task)
        
        return {'rot': rots, 'mic_pose': mic_poses, 'source_pose': source_poses}
    