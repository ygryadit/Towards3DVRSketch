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

```shell
# shape mode
python data_prep/gen_pointcloud.py -- $DATA_DIR/shape $DATA_DIR/shape/point shape
# sketch mode
python data_prep/gen_pointcloud.py -- $DATA_DIR/curve_network $DATA_DIR/curve_network/point sketch
python data_prep/gen_pointcloud.py -- $DATA_DIR/synthetic_sketch $DATA_DIR/synthetic_sketch/point sketch
python data_prep/gen_pointcloud.py -- $DATA_DIR/human_sketch $DATA_DIR/human_sketch/point sketch
```
### View rendering

> For both 3D shapes and 3D sketches we experiment with two types of rendering styles: Phong Shading and depth maps. For 3D sketches we represent each line as a 3D tube.

We use Blender to render views, so the first step is to replace [```blender_path```](data_prep/run_render_img.py#L6) with your local Blender path. 

There are 4 arguments for data_prep/run_render_img.py :

- $object_dir: directory path of your obj files
- $save_dir
- $data_type: 
    - `sketch`: sketch/curve metwork
    - `shape`: shape
- $render_type: 
    - `Phong`: Phong Shading
    - `depth`: Depth maps

```shell
python data_prep/run_render_img.py -- $object_dir $save_dir $data_type $render_type

# eg. rendering depth views for human sketches: 
python data_prep/run_render_img.py -- $DATA_DIR/human_sketch  $DATA_DIR/human_sketch/depth_view sketch depth
```

## Training

Run train_triplet.py  by:

```shell
python train_triplet.py \
    -epoch 300 \
    -batch_size 12 \
    -learning_rate 1e-2 \
    -margin 0.3 \
    -data_dir $DATA_DIR\
    --list_file $LIST_FILE_PATH \
```

## Evaluation

Run evaluation.py  by:

```shell
python evaluation.py \
    -epoch 300 \
    -batch_size 12 \
    -model_path $MODEL_PATH \
    -data_dir $DATA_DIR\
    --list_file $LIST_FILE_PATH \
```

You can easily adapt the evaluation code for inference only by:

```python
print('start argsort')
result = np.argsort(dist, axis=1) # the ranking of retrieved shapes
```
## Acknowledgements

Our work are based on several useful papers and projects:

- point_cloud_utils: https://github.com/fwilliams/point-cloud-utils
