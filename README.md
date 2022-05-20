
# Towards3DVRSketch
The code for the paper: 
```
"Towards 3D VR-Sketch to 3D Shape Retrieval"   
Ling Luo, Yulia Gryaditskaya, Yongxin Yang, Tao Xiang, Yi-Zhe Song
Proceedings of International Conference on 3D Vision (3DV), 2020
```
Project page: https://tinyurl.com/3DSketch3DV

> :tada: Important Update: We have published the first fine-grained human sketch dataset at https://cvssp.org/data/VRChairSketch/ for _Fine-Grained VR Sketching: Dataset and Insights_ on 3DV 2021.

# Description
The repository provides the code for synthetic sketch generation and the evaluated deep models.

# Synthetic sketch generation

## 1. Convert to manifold shapes
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
## 3. Synthetic sketch generation

### Dependencies
* libigl https://libigl.github.io/
* pyknotid https://pyknotid.readthedocs.io/en/latest/
* similaritymeasures 

```
pip install pyknotid
pip install similaritymeasures
```

### Step 1: Aggregation (C++)
To compile the code in SyntheticSketches/merge_lines, please see the README in SyntheticSketches/merge_lines

```shell
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

```shell
python SyntheticSketches/disturb_3d.py folder_netwroks folder_save
```
`folder_netwroks`
is a path to the networks from the previous step or generated with FlowRep;
`folder_save`
the path where to save the synthetic sketches.

# Dataset

This dataset includes .obj files and train/vallidation/test partition of:
- 3956 shapes + curve network + synthetic sketch from ModelNet10
- 167 human sketches from chair and bathtub classes of ModelNet10

Download link: [Google Drive][1]

# Deep models
The deep models and their usage is described in the subfolder: '3DSketchRetrieval'

# Contact information
Ling Luo: ling.rowling.luo@gmail.com


[1]: https://drive.google.com/file/d/1FkKZfWt7O4xMy4ir5kCYcmwZLPk1uBcZ/view?usp=sharing