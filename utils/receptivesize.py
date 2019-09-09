import os, sys, math
import os.path as osp
import pprint
from collections import OrderedDict
import pickle as pk
progpath = os.path.dirname(os.path.realpath(__file__))          # /home/luowei/Codes/feasc-msc
sys.path.append(progpath)

import torch
import torch.nn as nn


#################### model zoo
modelzoopath = "/home/luowei/Codes/pymodels"
# modelzoopath = "/vulcan/scratch/cswluo/Codes/pymodels"
sys.path.append(osp.dirname(modelzoopath))
import pymodels


#################### import modules in the current directory
import mymodels
import modellearning


#### model params
num_classes = 200
nparts = 2
seflag = True

model = mymodels.feasc50(num_classes=num_classes, nparts=nparts, seflag=seflag)

# for name, param in model.named_parameters():
#     print(name, '--->', param.size())

# for name, module in model.named_modules():
#     print(name, '--->', module)


print("\n==========================================================\n")

def reseq(module, layer_size, name=None, seqnum=None):
    if isinstance(module, nn.Sequential):
        reseq(module, layer_size, name, seqnum)
    else:
        for m in module.named_children():
            if isinstance(m[-1], (nn.Conv2d, nn.MaxPool2d)):
                kernel_size = m[-1].kernel_size[0]
                stride_size = m[-1].stride[0]
                padding_size = m[-1].padding[0]
                subname = m[0]
                print(name+'_'+str(seqnum)+'_'+subname, kernel_size, stride_size, padding_size)
                layer_size[name+'_'+str(seqnum)+'_'+subname] = [kernel_size, stride_size, padding_size]

layer_size = OrderedDict()
# this will not print the model itself
for name, module in model.named_children():
    # print(name, '--->', module)
    if isinstance(module, nn.Sequential):
        for i in range(len(module)):
            reseq(module[i], layer_size, name, i)

    if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = module.kernel_size
        stride_size = module.stride
        padding_size = module.padding
        print(name, kernel_size, padding_size, stride_size)
        if isinstance(module, nn.Conv2d):
            layer_size[name] = [kernel_size[0], stride_size[0], padding_size[0]]
        else:
            layer_size[name] = [kernel_size, stride_size, padding_size]

# pprint.pprint(layer_size)
# for key, value in layer_size.items():
#     print(key, value)

def outFromIn(conv, layerIn):
    n_in = layerIn[0]   # input feature dimension
    j_in = layerIn[1]   # jumps
    r_in = layerIn[2]   # receptive field
    start_in = layerIn[3]
    k = conv[0]     # kernel size
    s = conv[1]     # strides
    p = conv[2]     # padding

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k        # the total actual padding size
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3]))


layerInfos = []
if __name__ == "__main__":
    imgsize = 224
    r_in = 1
    s_in = 1
    start_in = 0
    currentLayer = [imgsize, s_in, r_in, start_in]
    layerInfos.append(currentLayer)
    printLayer(currentLayer, "input image")


    for key, value in layer_size.items():
        currentLayer = outFromIn(value, currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, key)
    print("------------------------")

    pprint.pprint(layerInfos)


