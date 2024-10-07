# Sound Spaces Data

## Visual Data Generation 
To train the NeRF part of NeRAF we use Habitat Sim to generate the visual data. More details on the motivation behind this can be found in the Appendix of the paper.

We installed Habitat Sim via the Sound Spaces 2.0 installation. You can follow the instructions [here](https://github.com/facebookresearch/sound-spaces).


We provide for each scene:
- simulator parameters in *scene_SimParams.json*
- agent position and orientation used for training in *scene_Train.pkl*
- agent position and orientation used for evaluation in *scene_Test.pkl*

The pkl files are in the following format:
```
{ (agent_pose_1, agent_rot_degree_1): {
    'Quaternion': , 
    'Position': , 
    }, 
    (agent_pose_2, agent_rot_degree_2): {
    'Quaternion': ,
    'Position': ,
    },
    ...
}
```
To generate the visual data, you can use *generate_vision.ipynb* script.

Note that: 
- The axis and quaternion convention are Habitat Sim: [x, y, z] = [right, up, backward]. Quaternion = [w, x, y, z]. More details can be found in Habitat Sim documentation. 
- The camera is located 1.5m above the agent position. 


## Audio Data 
We use Sound Spaces pre-generated binaural RIRs. They have been recorded in the navigable area of the scene.
You can download them [here](https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md).

Following previous works, we use 90% of them for training and 10% for evaluation.
We provide the split file for each scene in each scene folder.

We also provide a notebook, *process_audio.ipynb*, to pre-process the RIRs into magnitude STFTs at 22.05kHZ.
