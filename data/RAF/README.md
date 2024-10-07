# RAF Data

For both rooms, we rely on images-jpeg-1k from cameras 20 to 25. You can download the images from [here](https://github.com/facebookresearch/EyefulTower).
For the audio data, you can download the split and room impulse responses from [here](https://github.com/facebookresearch/real-acoustic-fields).

We use Metashape XML files to obtain the transform.json file needed for nerfstudio. 
Please refer to the [Nerfstudio Metashape documentation](https://docs.nerf.studio/quickstart/custom_dataset.html#metashape) for more information.

For each room:
1. Download the images from the link above.

2. Keep only images from cameras 20 to 25. The folder should have the following structure:
```
├── images-jpeg-1k
│   ├── 20_DSC0001.jpg
│   ├── ...
│   ├── 25_DSCXXX.jpg
```

3. Run the following command with the given XML file to generate the transform.json file:
```
ns-process-data metashape --data {data directory} --xml {xml file} --output-dir {output directory}
```
This command will copy in the output directory the images renamed in nerfstudio format, downsampled versions of the images and the generated transform.json file. 

4. We have subsampled transform.json to randomly keep one third of the images. 

We provide the XML and the **final** transform.json files:
```
├── EmptyRoom
│   ├── EmptyRoom_20-25.xml
│   ├── transforms.json
├── FurnishedRoom
│   ├── FurnishedRoom_20-25.xml
│   ├── transforms.json
```

In the end, you should have the following structure:
```
├── EmptyRoom
│   ├── images
│   │   ├── frame_00001.jpg
│   │   ├── ...
│   ├── data
│   ├── metadata
│   │   ├── data-split.json
│   ├── transforms.json
├── FurnishedRoom
│   ├── images
│   │   ├── frame_00001.jpg
│   │   ├── ...
│   ├── data
│   ├── metadata
│   │   ├── data-split.json
│   ├── transforms.json
```

With 
- images: the folder generated by the command ns-process-data
- data: the folder containing RAF data (audio)
- metadata: the folder containing RAF data-split.json file
- transforms.json: the file generated by the command ns-process-data downsampled i.e., the one we provide. 


