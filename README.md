# NeRAF
This repository is the official implementation of NeRAF: 3D Scene Infused Neural Radiance and Acoustic Field.
NeRAF is a novel method that learns neural radiance and acoustic field. 

The code is written in Python and uses the NerfStudio framework.

## File Structure
```
├── NeRAF
│   ├── __init__.py
│   ├── NeRAF_config.py
│   ├── NeRAF_pipeline.py 
│   ├── NeRAF_model.py 
│   ├── NeRAF_field.py 
│   ├── NeRAF_datamanger.py 
│   ├── NeRAF_resnet3d.py
│   ├── NeRAF_evaluator.py
├── pyproject.toml
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
ns-train NeRAF --data [PATH]
```

## Evaluation
To evaluate my model on eval set, run the following command:
```
ns-eval --load-config [CONFIG_PATH] --output-path [OUTPUT_PATH] --render-output-path [RENDER_OUTPUT_PATH]
```
More informations are provided in Nerfstudio documentation. 
ns-eval will save rendered images and audio in the output path.


## Resume training 
To resume training, run the following command:
```
ns-train NeRAF --data [PATH] --load-dir [MODEL_PATH]
```
More informations are provided in Nerfstudio documentation. 

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg