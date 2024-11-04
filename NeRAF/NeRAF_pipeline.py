"""
NeRAF Pipeline
"""

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfstudio.pipelines.base_pipeline import Pipeline

import numpy as np
import os
import cv2


@dataclass
class NeRAFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeRAFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    audio_datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    vision_model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""
    audio_model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""

    start_step_audio: int = 2000
    """specifies the step to start training the audio model"""

    save_eval_audio_path: str = None


class NeRAFPipeline(VanillaPipeline):
    """NeRAF Pipeline.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: NeRAFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        # set same data dir for audio and vision
        config.audio_datamanager.data = config.datamanager.data

        self.save_eval_audio_path = config.save_eval_audio_path
        
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.audio_datamanager: DataManager = config.audio_datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

        self.datamanager.to(device)
        self.audio_datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.vision_model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.audio_model = config.audio_model.setup(
            scene_box=self.audio_datamanager.train_dataset.scene_box,
            num_train_data=len(self.audio_datamanager.train_dataset),
            device=device,
        )

        # Retrieve spatial distortion from vision model for grid sampling
        #print('Givin SD to audio model')
        self.audio_model.spatial_distortion = self.model.field.module.spatial_distortion

        self.audio_model.to(device)

        self.audio_model.set_eval_data(self.audio_datamanager.eval_dataset[0]['source_pose'], self.audio_datamanager.eval_dataset[0]['mic_pose'], self.audio_datamanager.eval_dataset[0]['rot'], self.audio_datamanager.eval_dataset[0]['data'])

        # Give to vision model the audio model for viewer
        self.model.audio_model = self.audio_model
        

        self.world_size = world_size
        if world_size > 1:
            raise NotImplementedError("Distributed training is not supported yet")
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.start_step_audio = self.config.start_step_audio

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        #updating a part of the grid
        if self.config.audio_model.use_grid:
            self.audio_model.query_grid_one_batch(step, self.model.field,
                                                renderer_rgb = self.model.renderer_rgb,
                                                batch_size=self.config.datamanager.train_num_rays_per_batch)

        if step > self.start_step_audio:
            _, batch_audio = self.audio_datamanager.next_train(step) # we don't need ray_bundle for audio model
            model_audio_outputs = self.audio_model.get_outputs(batch_audio) 
            # metrics_audio_dict = self.audio_model.get_metrics_dict(model_audio_outputs, batch_audio)
            metrics_audio_dict = {}  # we don't compute metrics for training to speedup
            loss_audio_dict = self.audio_model.get_loss_dict(model_audio_outputs, batch_audio, metrics_audio_dict)


            # we merge the loss and metrics dict
            for key in loss_audio_dict:
                loss_dict[key] = loss_audio_dict[key]

            for key in metrics_audio_dict:
                metrics_dict[key] = metrics_audio_dict[key]

        return model_outputs, loss_dict, metrics_dict
    
    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)


        if step > self.start_step_audio:
            _, batch_audio = self.audio_datamanager.next_eval(step)
            model_audio_outputs = self.audio_model.get_outputs(batch_audio)
            metrics_audio_dict = self.audio_model.get_metrics_dict(model_audio_outputs, batch_audio)
            loss_audio_dict = self.audio_model.get_loss_dict(model_audio_outputs, batch_audio, metrics_audio_dict)

            for key in loss_audio_dict:
                loss_dict[key] = loss_audio_dict[key]

            for key in metrics_audio_dict:
                metrics_dict[key] = metrics_audio_dict[key]

        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera,None, eval=True)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        if step > self.start_step_audio:
            camera_audio, batch_audio = self.audio_datamanager.next_eval_image(step)
            
            outputs_audio = self.audio_model.get_outputs_for_camera(camera_audio,None,batch_audio=batch_audio)
            metrics_audio_dict, images_audio_dict = self.audio_model.get_image_metrics_and_images(outputs_audio, batch_audio)

            for key in metrics_audio_dict:
                metrics_dict[key] = metrics_audio_dict[key]

            for key in images_audio_dict:
                images_dict[key] = images_audio_dict[key]

        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        metrics_dict_audio_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        if self.audio_datamanager.eval_dataset.mode != "inference":
            self.audio_datamanager.eval_dataset.mode = 'eval_image' 
        num_stft = len(self.audio_datamanager.eval_dataset)
        nb_im = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera,obb_box=None, eval=True)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, im = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    im = im["img"].detach().cpu().numpy()
                    str_nb_im = str(nb_im).zfill(5)
                    #save im into output_path
                    im_path = os.path.join(output_path, f"eval_{str_nb_im}.png")
                    im = (im * 255).astype(np.uint8)

                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(im_path, im)
                    nb_im += 1
                    # raise NotImplementedError("Saving images is not implemented yet")
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

            # evaluate audio
            if output_path is not None: 
                step = self.start_step_audio + 1
            if step > self.start_step_audio:
                task = progress.add_task("[green]Evaluating all eval stft...", total=num_stft)
                output_eval = []
                nb_stft = 0
                for i in range(num_stft):
                    # time this the following line
                    inner_start = time()

                    #should be done in datamanager...
                    batch = self.audio_datamanager.eval_dataset[i]

                    outputs = self.audio_model.get_outputs_for_camera(None,None,batch_audio=batch)
                    
                    metrics_dict, _ = self.audio_model.get_image_metrics_and_images(outputs, batch)

                    if self.save_eval_audio_path is not None:
                        #save stft
                        batch_output = batch
                        batch_output["pred"] = outputs["raw_output"].detach().cpu().numpy()
                        if not os.path.exists(os.path.join(self.save_eval_audio_path, str(step))):
                            os.makedirs(os.path.join(self.save_eval_audio_path, str(step)))
                        np.save(os.path.join(self.save_eval_audio_path, str(step), f"eval_{i}.npy"), batch_output)        

                    if output_path is not None:
                        audio2save = outputs['raw_output'].permute(1,2,0).detach().cpu().numpy()
                        # save as a wav file 
                        str_nb_stft = str(nb_stft).zfill(5)
                        audio_path = os.path.join(output_path, f"eval_{str_nb_stft}.npy")
                        np.save(audio_path, audio2save)
                        nb_stft += 1

                    num_rays = batch['data'].shape[-1]
                    assert "num_rays_per_sec_audio" not in metrics_dict
                    metrics_dict["num_rays_per_sec_audio"] = (num_rays / (time() - inner_start))
                    fps_str = "fps_audio"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec_audio"] / (num_rays))

                    # put metrics_dict on cpu and detach
                    for key in metrics_dict:
                        if isinstance(metrics_dict[key], torch.Tensor):
                            metrics_dict[key] = metrics_dict[key].cpu().detach()
                        #metrics_dict[key] = metrics_dict[key].cpu().detach()

                    metrics_dict_audio_list.append(metrics_dict)
                    progress.advance(task)
                
                if self.audio_datamanager.eval_dataset.mode != "inference":
                    self.audio_datamanager.eval_dataset.mode = 'eval'
   
                
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        if step > self.start_step_audio:
            metrics_dict_audio = {}
            for key in metrics_dict_audio_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor([metrics_dict_[key] for metrics_dict_ in metrics_dict_audio_list])
                    )
                    metrics_dict_audio[key] = float(key_mean)
                    metrics_dict_audio[f"{key}_std"] = float(key_std)
                else:
                    metrics_dict_audio[key] = float(
                        torch.mean(torch.tensor([metrics_dict_[key] for metrics_dict_ in metrics_dict_audio_list]))
                    )
                    

            for key in metrics_dict_audio:
                metrics_dict[key] = metrics_dict_audio[key]


        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.audio_model.update_to_step(step)
        
        grid = state.pop("audio_model.grid") # should be replaced by saved features
        
        self.load_state_dict(state)
        
        self.audio_model.grid = grid.to(self.device)

        # Retrieve spatial distortion from vision model for grid samplint
        self.audio_model.spatial_distortion = self.model.field.module.spatial_distortion

        # For debug, to be changed later to accomodate viewer camera
        self.audio_model.set_eval_data(self.audio_datamanager.eval_dataset[0]['source_pose'], self.audio_datamanager.eval_dataset[0]['mic_pose'], self.audio_datamanager.eval_dataset[0]['rot'], self.audio_datamanager.eval_dataset[0]['data'])

        # Give to vision model the audio model for viewer output
        self.model.audio_model = self.audio_model

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks

        ### TODO Audio callbacks to add here
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        audio_model_params = self.audio_model.get_param_groups()

        audio_model_params["audio_fields"].extend(model_params["fields"]) # Backprop on vision too
        # model_params["fields"].extend(audio_model_params["audio_fields"])

        return {**datamanager_params, **model_params, **audio_model_params}

    def state_dict(self):
        state_dict = super().state_dict()
        # Save the grid: in practice this is not necessary but usefull for debug
        # will be replaced by saved features
        state_dict["audio_model.grid"] = self.audio_model.grid
        return state_dict