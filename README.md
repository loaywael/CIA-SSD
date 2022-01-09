## CIA-SSD: Confident IoU-Aware Single Stage Object Detector From Point Cloud (AAAI 2021) [[Paper]](https://arxiv.org/abs/2012.03015)

Single-stage object detector from point cloud on KITTI Benchmark, running with 32FPS.

**Authors**: [Wu Zheng](https://github.com/Vegeta2020), Weiliang Tang, Sijin Chen, [Li Jiang](https://github.com/llijiang), Chi-Wing Fu.

## AP on KITTI Dataset


My Training results on [Car, Pedestrian, Cyclist] for 75 epochs
```
2022-01-09 07:46:40,337 - INFO - Evaluation official: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.74, 89.04, 80.86
bev  AP:90.48, 87.82, 80.11
3d   AP:88.44, 77.22, 69.65
aos  AP:90.65, 88.36, 80.11
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.74, 89.04, 80.86
bev  AP:90.85, 89.92, 89.41
3d   AP:90.84, 89.82, 89.19
aos  AP:90.65, 88.36, 80.11
pedestrian AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:53.59, 45.82, 39.16
bev  AP:54.56, 46.45, 44.55
3d   AP:50.41, 42.67, 36.68
aos  AP:42.71, 36.07, 30.81
pedestrian AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:53.59, 45.82, 39.16
bev  AP:67.09, 57.90, 50.32
3d   AP:67.09, 57.90, 50.32
aos  AP:42.71, 36.07, 30.81
cyclist AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:86.85, 75.71, 73.78
bev  AP:83.81, 69.78, 64.44
3d   AP:82.20, 66.86, 62.40
aos  AP:86.64, 74.02, 72.06
cyclist AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:86.85, 75.71, 73.78
bev  AP:91.63, 71.56, 69.69
3d   AP:91.63, 71.56, 69.69
aos  AP:86.64, 74.02, 72.06
```

Val Split (11 recall points) using the author [pretrained-model](https://drive.google.com/file/d/1SElYNQCsr4gctqLxmB6Fc4t7Ed8SgBgs/view?usp=sharing) on cars only:
```
Car  AP:98.85, 90.20, 89.58
bev  AP:90.51, 88.86, 87.95
3d   AP:90.00, 79.86, 78.83
aos  AP:98.77, 89.99, 89.24
Car  AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:98.85, 90.20, 89.58
bev  AP:98.92, 90.29, 89.81
3d   AP:99.00, 90.22, 89.70
aos  AP:98.77, 89.99, 89.24
```
trained on the train split (3712 samples)
Test Split: [Submission link](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=b4e17f75f5baa917c4f250e832aace71682c3a84)


## Pipeline

![pipeline](https://github.com/Vegeta2020/CIA-SSD/blob/master/pictures/pipeline.png)
The pipeline of our proposed Confident IoU-Aware Single-Stage object Detector (CIA-SSD). First, we encode the input point cloud (a) with a sparse convolutional network denoted by SPConvNet (b), followed by our spatial-semantic feature aggregation (SSFA) module (c) for robust feature extraction, in which an attentional fusion module (d) is adopted to adaptively fuse the spatial and semantic features. Then, the multi-task head (e) realizes the object classification and localization, with our introduced confidence function (CF) for confidence rectification. In the end, we further formulate the distance-variant IoU-weighted NMS (DI-NMS) for post-processing.

## Installation

```bash
$ git clone https://github.com/Vegeta2020/CIA-SSD.git
$ cd ./CIA-SSD/det3d/core/iou3d
$ python setup.py install
$ cd ./CIA-SSD
$ python setup.py build develop
```
Please follow Det3D for installation of other [related packages](https://github.com/poodarchu/Det3D/blob/master/INSTALLATION.md) and [data preparation](https://github.com/poodarchu/Det3D/blob/master/GETTING_STARTED.md).

## Train and Eval

Configure the model in
```bash
$ /CIA-SSD/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py
```

Please use our code to generate ground truth data:
```bash
$ python ./CIA-SSD/tools/create_data.py
```

Train the CIA-SSD:
```bash
$ cd ./CIA-SSD/tools
$ python train.py  # Single GPU
$ python -m torch.distributed.launch --nproc_per_node=4 train.py   # Multiple GPU
```

Evaluate the CIA-SSD:
```bash
$ cd ./CIA-SSD/tools
$ python test.py
```

## Citation
If you find this work useful in your research, please star our repository and consider citing:
```
@inproceedings{zheng2020ciassd,
  title={CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud},
  author={Wu Zheng, Weiliang Tang, Sijin Chen, Li Jiang, Chi-Wing Fu},
  booktitle={AAAI},
  year={2021}
}
```


## License
This codebase is released under the Apache 2.0 license.

## Acknowledgement
Our code are mainly based on [Det3D](https://github.com/poodarchu/det3d), thanks for their contributions! We also thank for the reviewers's valuable comments on this paper.


## Contact
If you have any question regarding ros support, multi-target training, or 360-deg support feel free to contact me (hythamconn1@gmail.com)
I highly recommend asking the main author(zheng-w10@foxmail.com) about new suguestions, model architecture or any other issue regarding the originalrepo 
