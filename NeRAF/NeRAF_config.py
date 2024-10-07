"""
NeRAF Config
"""

from __future__ import annotations

from NeRAF.NeRAF_pipeline import NeRAFPipelineConfig, NeRAFPipeline

from NeRAF.NeRAF_model import NeRAFVisionModelConfig, NeRAFAudioModelConfig

from NeRAF.NeRAF_dataparser import SoundSpacesDataParserConfig, RAFDataParserConfig
from NeRAF.NeRAF_datamanager import SoundSpacesDataManagerConfig, RAFDataManagerConfig


from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from pathlib import Path
import os 

# Parameters to change for each experiment ----
dataset = 'RAF' # or 'SoundSpaces'
scene = 'FurnishedRoom'

if "NeRAF_dataset" in os.environ:
    dataset = os.environ["NeRAF_dataset"]
if "NeRAF_scene" in os.environ:
    scene = os.environ["NeRAF_scene"]

if dataset == 'SoundSpaces':
    fs = 22050
    max_len_SoundSpaces = {'office_4': 78, 'room_2': 84, 'frl_apartment_2': 107, 'frl_apartment_4': 103, 'apartment_2': 86, 'apartment_1': 101}
    max_len = max_len_SoundSpaces[scene]
    base_dir = '../data/SoundSpaces'
    eval_mode_vision = 'filename'
    datamanager = SoundSpacesDataManagerConfig(train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            fs=fs,
            max_len=max_len,
            dataparser=SoundSpacesDataParserConfig())
else:
    fs = 48000
    max_len = 0.32 
    base_dir = '../data/RAF'
    eval_mode_vision = 'fraction'
    datamanager = RAFDataManagerConfig(train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            fs=fs,
            max_len=max_len,
            dataparser=RAFDataParserConfig())

experiment_name= scene + '_NeRAF'
output_dir = "./outputs" # path to save checkpoints and logs
eval_save_dir = None # give path to save rendered eval STFT (visualization)
start_step_audio = 2000 # number of iterations NeRF has been trained before starting audio training
# ----
 
NeRAF_method = MethodSpecification(
    config=TrainerConfig(
    method_name="NeRAF",
    experiment_name=experiment_name, 
    steps_per_eval_batch=10000,
    steps_per_eval_image=10000,
    steps_per_eval_all_images=10000,
    steps_per_save=20000,
    save_only_latest_checkpoint=False,
    max_num_iterations=400001,
    mixed_precision=True,
    data=Path(os.path.join(base_dir, scene)), 
    output_dir=Path(output_dir),
    pipeline=NeRAFPipelineConfig(
        datamanager=ParallelDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                eval_mode=eval_mode_vision,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            images_on_gpu=True,
            masks_on_gpu=True,
        ),
        audio_datamanager=datamanager,

        vision_model=NeRAFVisionModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
        audio_model=NeRAFAudioModelConfig(
            dataset=dataset,
            use_grid=True,
            grid_step=1/128,
            N_features=1024, 
            use_multiple_viewing_directions=True,
            loss_factor=1e-3, # weight the audio loss
            W_field=512,
            N_freq_stft=257,
            fs = fs,
            criterion="SC+SLMSE", # or SC+SLL1
            max_len=max_len,
        ),
        start_step_audio=start_step_audio,
        save_eval_audio_path=eval_save_dir,
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "audio_fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-8, max_steps=1000000+start_step_audio, warmup_steps=start_step_audio),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
    # vis="viewer",
    ),
    description="NeRAF method.",
    
)