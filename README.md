
# Towards3DVRSketch
The code for the paper: 
```
"Towards 3D VR-Sketch to 3D Shape Retrieval"   
Ling Luo, Yulia Gryaditskaya, Yongxin Yang, Tao Xiang, Yi-Zhe Song
Proceedings of International Conference on 3D Vision (3DV), 2020
```
# Description
The repository provides the code for synthetic sketch generation and the evaluated deep models.

# Synthetic sketch generation
## 1. Conver to manifold shapes
Since many shapes in the publicly available datasets are not manifold shapes, we first recommend preprocessign shapes with this method:
https://github.com/hjwdzh/ManifoldPlus

## 2. Extract curve networks
To extract the curve netwrok we use the auhtors implementation of this paper:
https://www.cs.ubc.ca/labs/imager/tr/2017/FlowRep/
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

## 3. Synhteic sketch generation

# Deep models
