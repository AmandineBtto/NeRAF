# Visual Data Generation

To train the NeRF part of NeRAF we use Habitat Sim to generate the visual data. 
More details can be found in the Supplementary Material of the paper.

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

Note that: 
- The axis and quaternion convention are Habitat Sim: [x, y, z] = [right, up, backward]. Quaternion = [w, x, y, z]. More details can be found in Habitat Sim documentation. 
- The camera is located 1.5m above the agent position. 