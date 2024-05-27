"""
NeRAF Config
"""

from __future__ import annotations

from NeRAF.NeRAF_pipeline import NeRAFPipelineConfig,NeRAFPipeline

from NeRAF.NeRAF_model import NeRAFVisionModelConfig, NeRAFAudioModelConfig

from NeRAF.NeRAF_datamanager import NeRAFDataManagerConfig


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

start_step_audio = 2000 # number of iterations NeRF has been trained before starting audio training
fs = 22050

# Parameters to change for each experiment --
experiment_name= "office4_NeRAF"
output_dir = "./outputs" # path to save checkpoints and logs
eval_save_dir = None # give path to save rendered eval STFT (visualization)
# office4: 78, room2: 84, frl apt 2: 107, frl apt 4: 103, apt 2: 86, apt 1: 101
max_len = 78
# --
 
NeRAF_method = MethodSpecification(
    config=TrainerConfig(
    method_name="NeRAF",
    experiment_name=experiment_name, 
    steps_per_eval_batch=5000,
    steps_per_eval_image=5000,
    steps_per_eval_all_images=25000,
    steps_per_save=25000,
    save_only_latest_checkpoint=True,
    max_num_iterations=500001,
    mixed_precision=True,
    data=Path('./data/Replica/office_4'), # to be changed depending of the scene
    output_dir=Path(output_dir),
    pipeline=NeRAFPipelineConfig(
        datamanager=ParallelDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                eval_mode="filename",
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            images_on_gpu=True,
            masks_on_gpu=True,
        ),
        audio_datamanager=NeRAFDataManagerConfig(
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            fs=fs,
            max_len=max_len,
            # hop_len=128,
        ),
        vision_model=NeRAFVisionModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
        audio_model=NeRAFAudioModelConfig(
            use_grid=True,
            grid_step=1/128,
            use_multiple_viewing_directions=True,
            loss_factor=1e-3,
            W_field=512,
            N_freq_stft=257,
            fs = fs,
            criterion="SC+SLMSE",
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