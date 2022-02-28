# Training/evaluating the network
---
## Prerequisites

- Pytorch 1.7.0
- point_cloud_utils: https://github.com/fwilliams/point-cloud-utils
- tensorboardX (optional)


## Data Preparation

Once downloaded the dataset to local path $DATA_DIR, we need to process those objs files into pointcloud for point-based methods or render views for view-based methods.

### Point sampling

We need to sample 10000 points from each obj file by running data_prep/gen_pointcloud.py :
```
python data_prep/gen_pointcloud.py -- $OBJ_DIR $SAVE_DIR $DATA_TYPE
```
There are 2 choices for DATA_TYPE: 'shape' and 'sketch'. 
- 'shape' mode samples points from faces, which is used to generate pointcloud from shapes with Monte-Carlo sampling.
- 'sketch' mode samples points from edges, which is designed for generating pointcloud from curve network and human/synthetic sketches with equidistant sampling.

```
# shape mode
python data_prep/gen_pointcloud.py -- $DATA_DIR/shape $DATA_DIR/shape/point shape
# sketch mode
python data_prep/gen_pointcloud.py -- $DATA_DIR/curve_network $DATA_DIR/curve_network/point sketch
python data_prep/gen_pointcloud.py -- $DATA_DIR/synthetic_sketch $DATA_DIR/synthetic_sketch/point sketch
python data_prep/gen_pointcloud.py -- $DATA_DIR/human_sketch $DATA_DIR/human_sketch/point sketch
```
### View rendering

> For both 3D shapes and 3D sketches we experiment with two types of rendering styles: Phong Shading and depth maps. For 3D sketches we represent each line as a 3D tube.

We use Blender to render views, so the first step is to replace 'blender_path' in line 5 of data_prep\run_model_mesh.py with your local Blender path. 

- $data_type: 
    - `sketch`: sketch/curve metwork
    - `shape`: shape
- $render_type: 
    - `Phong`: Phong Shading
    - `depth`: Depth maps

```
python data_prep/run_render_img.py -- $object_dir $save_dir $data_type $render_type

# eg. rendering depth views for human sketches: 
python data_prep/run_render_img.py -- $DATA_DIR/human_sketch  $DATA_DIR/human_sketch/depth_view sketch depth
```

### Training

Change the ROOT_DIR in config.py to your data path. Run train_triplet.py  by:

```
python train_triplet.py \
    -epoch 300 \
    -batch_size 12 \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --save_freq 10 \
    --log_freq 10 \
    --margin 0.3 \
    --save_name $SAVE_NAME \
    --list_file $LIST_FILE_PATH \
    --train
```

### Evaluation

### Acknowledgements

Our work are based on several useful papers and projects:

- point_cloud_utils: https://github.com/fwilliams/point-cloud-utils
