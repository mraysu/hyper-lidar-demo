# Demo Software for [LidarSegHD](https://github.com/DarthIV02/LidarSegHD)

# Code adapted from [API for SemanticKITTI](https://github.com/PRBonn/semantic-kitti-api)

This repository contains helper scripts to open, visualize, process, and 
evaluate results for point clouds and labels from the SemanticKITTI dataset.

- Link to original [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) Dataset
- Link to [SemanticKITTI dataset](http://semantic-kitti.org/).
- Link to SemanticKITTI benchmark [competition](http://semantic-kitti.org/tasks.html).

---
##### Example of 3D pointcloud from sequence 13:
<img src="https://image.ibb.co/kyhCrV/scan1.png" width="1000">

---

#TODO: Format Citations, clean out documentation

## Usage:
````
python3 visualize.py --help
````

For default parameters:
````
python3 visualize.py --dataset DATASET
````
To specify label config .yml: (defaults to config/semantic-kitti.yaml)
````
python3 visualize.py --dataset DATASET --config CONFIG.YML
````
To see data load and visualization Time:
````
python3 visualize.py --dataset DATSET --print_data
````
