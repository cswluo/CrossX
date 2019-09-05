# CrossX

This is PyTorch implementation of our ICCV 2019 paper "Cross-X Learning for Fine-Grained Visual Categorization". We experimented on 5 fine-grained benchmark datasets --- NABirds, CUB-200-2011, Stanford Cars, Stanford Dogs, and VGG-Aircraft. You should first download these datasets from their project homepages before runing CrossX.


## Appoach

![alt text](https://github.com/cswluo/CrossX/blob/crossx/crossx.png)

## Implementation

A "x-imdb.py" is provided for each dataset to generate Python pickle files, which are then used to prepare train/val/trainval/test data. Run "x-imdb.py" in the folder of your dataset to generate corresponding pickle file (imdb.pkl) should be the very first step.

- demo.py is used to train your own CrossX model from scratch.

- prediction.py outputs classification accuracy by employing pretrained CrossX models.   

Due to the random generation of train/val/test data on some datasets, the classification accuracy may have a bit fluctuation but it should be in a reasonable range.

The pretrained CrossX models can be download from [HERE](https://pan.baidu.com/s/1803G5v0KDU0B_NS62Ril3A). If you plan to train your own CrossX model from scratch by using the SENet backbone, you should download the pretrained SENet-50 weights from [HERE](https://pan.baidu.com/s/1k6NaffqmbakH9Vng-CLxlg).

## Results

|              | CrossX-SENet-50 | CrossX-ResNet-50 |
|:-------------|:---------------:|:----------------:|
|NABirds       |86.4%            |86.2%             |
|CUB-200-2011  |87.5%            |87.7%             |
|Stanford Cars |94.5%            |94.6%             |
|Stanford Dogs |88.2%            |88.9%             |
|VGG-Aircraft  |92.7%            |92.6%             |


## Citation

If you use CrossX in your research, please cite the paper:
```
@inproceedings{luowei@19iccv,
author = {Wei Luo and Xitong Yang and Xianjie Mo and Yuheng Lu and Larry S. Davis and Ser-Nam Lin},
title = {Cross-X learning for fine-grained visual categorization},
booktitle = {ICCV},
year = {2019},
}
```
