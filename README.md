# NeRAF: 3D Scene Infused Neural Radiance and Acoustic Fields [Accepted at ICLR 2025]

<p align="center" style="margin: 2em auto;">
    <a href='https://amandinebtto.github.io/NeRAF' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/NeRAF-Project_page-orange?style=flat&logo=github&logoColor=orange' alt='Project Page'></a>
    <a href='https://arxiv.org/abs/2405.18213'><img src='https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arXiv&logoColor=green' alt='Paper'></a>
</p>

This repository is the official implementation of NeRAF: 3D Scene Infused Neural Radiance and Acoustic Fields.
NeRAF is a novel method that learns neural radiance and acoustic field. 

## File Structure
```
├── NeRAF
│   ├── __init__.py
│   ├── NeRAF_config.py
│   ├── NeRAF_pipeline.py 
│   ├── NeRAF_model.py 
│   ├── NeRAF_field.py 
│   ├── NeRAF_datamanager.py 
│   ├── NeRAF_dataparser.py 
│   ├── NeRAF_dataset.py 
│   ├── NeRAF_helpers.py 
│   ├── NeRAF_resnet3d.py
│   ├── NeRAF_evaluator.py
├── pyproject.toml
├── data
│   ├── RAF
│   ├── SoundSpaces
```

## Requirements 
### Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd NeRAF/
pip install -e .
ns-install-cli
```

### Additional packages
Additional audio related packages must be added to the conda environment.
- librosa 
- pyroomacoustics
- torchaudio
- scipy

## Training
This code creates a new Nerfstudio method named "NeRAF-method". 
Change the configuration file NeRAF_config.py to set the parameters for the method.
The file already contains the default parameters used in the paper.

To train the models, run the command:
```
NeRAF_dataset=RAF NeRAF_scene=FurnishedRoom ns-train NeRAF 
```

## Evaluation & Weights
We provide the [weights](https://univpsl-my.sharepoint.com/:f:/g/personal/amandine_brunetto_minesparis_psl_eu/EnBwmOmNIUxNiTwKT_-eKhwBhtIlk5z5v6yWPXjCmnjsLw?e=CbQGZL) of the model for every scene in RAF and SoundSpaces datasets.
Note that if you train NeRAF from scratch you may obtain slightly different performances as the training is non-deterministic. This is why we averaged results on multiple runs in the paper. 

To evaluate the model on eval set, run the following command:
```
ns-eval --load-config [CONFIG_PATH to config.yml] --output-path [OUTPUT_PATH to out_name.json] --render-output-path [RENDER_OUTPUT_PATH to folder conainting rendered images and audio in the eval set]
```
More informations are provided in Nerfstudio documentation. 

ns-eval will save rendered images and audio in the render output path and give averaged metrics in the output path. 


## Resume training 
To resume training, run the following command:
```
ns-train NeRAF --load-dir [MODEL_PATH]
```
More informations are provided in Nerfstudio documentation. 

## Citation 
If you find this repository useful in your research, please consider giving a star and cite our paper by using the following: 

```
@article{brunetto2024neraf,
  title={NeRAF: 3D Scene Infused Neural Radiance and Acoustic Fields},
  author={Brunetto, Amandine and Hornauer, Sascha and Moutarde, Fabien},
  journal={arXiv preprint arXiv:2405.18213},
  year={2024}
}
```

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
