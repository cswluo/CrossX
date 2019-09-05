# CrossX
Coming soon ...

This is PyTorch implementation of our ICCV 2019 paper "Cross-X Learning for Fine-Grained Visual Categorization". We experimented on 5 fine-grained benchmark datasets --- NABirds, CUB-200-2011, Stanford Cars, Stanford Dogs, and VGG-Aircraft. A "x-imdb.py" is provided for each dataset to generate Python pickle files, which are then used to prepare train/val/trainval/test data.

--- demo.py is used to train your own CrossX model.

--- prediction.py outputs classification accuracy by employing pretrained CrossX models.   

Due to the random generation of train/val/test data on some datasets, the classification accuracy may have a bit fluctuation but it should be in a reasonable range.

The pretrained CrossX models can be download from here. If you want to train your CrossX model from scratch by using the SENet backbone, you can download the pretrained SENet-50 weights from here.


