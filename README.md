
# Towards3DVRSketch
The code for the paper: 
```
"Towards 3D VR-Sketch to 3D Shape Retrieval"   
Ling Luo, Yulia Gryaditskaya, Yongxin Yang, Tao Xiang, Yi-Zhe Song
Proceedings of International Conference on 3D Vision (3DV), 2020
```
Project page: https://tinyurl.com/3DSketch3DV
# Description
The repository provides the code for synthetic sketch generation and the evaluated deep models.

# Synthetic sketch generation
## 1. Conver to manifold shapes
Since many shapes in the publicly available datasets are not manifold shapes, we first recommend preprocessign shapes with this method:
https://github.com/hjwdzh/ManifoldPlus
```
@article{huang2020manifoldplus,
  title={ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups},
  author={Huang, Jingwei and Zhou, Yichao and Guibas, Leonidas},
  journal={arXiv preprint arXiv:2005.11621},
  year={2020}
}
```

## 2. Extract curve networks
To extract the curve netwrok we use the auhtors implementation of this paper:
https://www.cs.ubc.ca/labs/imager/tr/2017/FlowRep/
```
@article{59,
  author  = {Gori, Giorgio and Sheffer, Alla and Vining, Nicholas and Rosales, Enrique and Carr, Nathan and Ju, Tao},
  title   = {FlowRep: Descriptive Curve Networks for Free-Form Design Shapes},
  journal = {ACM Transaction on Graphics},
  year    = {2017},
  volume = {36},
  number = {4},
  doi = {http://dx.doi.org/10.1145/3072959.3073639},
  publisher = {ACM},
  address = {New York, NY, USA}
}
```
## 3. Syntheic sketch generation
### Dependencies
* libigl https://libigl.github.io/
### Step 1: Aggregation (C++)
To compile the code in SyntheticSketches/merge_lines, please see the README in SyntheticSketches/merge_lines
```
python SyntheticSketches/agrregate_network.py folder_netwroks folder_save executable_path
```
where
`folder_netwroks`
is a path to the networks generated with FlowRep;
`folder_save`
the path where to save the cleaned networks;
`executable_path`
the path to a compiled SyntheticSketches/merge_lines.

### Step 2: Aggregation & Distortion (Python)
```
python SyntheticSketches/disturb_3d.py folder_netwroks folder_save
```
`folder_netwroks`
is a path to the networks from the previous step or generated with FlowRep;
`folder_save`
the path where to save the synthetic sketches.
# Deep models
The deep models and their usage is described in the subfolder: 'deep_models'

# Contact information
Ling Luo: ling.rowling.luo@gmail.com
