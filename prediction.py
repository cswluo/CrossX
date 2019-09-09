import os, sys
import pickle as pk
import pdb

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo

from initialization import data_transform
from utils import imdb
progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)
import modellearning


""" user params """
datasetname = "vggaircraft" # 'cubbirds', 'nabirds', 'stdogs', 'stcars'
batchsize = 8
backbone = 'resnet' # or 'senet'
device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")



#################### model zoo
modelzoopath = "/home/luowei/Codes/pymodels"
sys.path.append(os.path.dirname(modelzoopath))
# import pymodels

##################### Dataset path
datasets_path = os.path.expanduser("/home/luowei/Datasets")
datasetpath = os.path.join(datasets_path, datasetname)


################### organizing data
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
data_transform = data_transform(datasetname)


testsplit = datasets.ImageFolder(os.path.join(datasetpath, 'test'), data_transform['test'])
testloader = torch.utils.data.DataLoader(testsplit, batch_size=batchsize, shuffle=False, num_workers=8)


datasplit_sizes = len(testsplit)
class_names = testsplit.classes
num_classes = len(class_names)

################################### constructing or loading model
if datasetname is 'stdogs' and backbone is 'senet':
    nparts = 3
else:
    nparts = 2   # number of parts you want to use for your dataset 

if backbone is 'senet':
    if datasetname in ['cubbirds', 'nabirds']:
        import mysenetmodelsmix as crossxmodel
        model = crossxmodel.senet50(num_classes=num_classes, nparts=nparts)
    else:
        import mysenetmodelsavg as crossxmodel
        model = crossxmodel.senet50(num_classes=num_classes, nparts=nparts)
elif backbone is 'resnet':
    if datasetname in ['cubbirds', 'nabirds']:
        import myresnetmodelsmix as crossxmodel
        model = crossxmodel.resnet50(num_classes=num_classes,  nparts=nparts)
    else:
        import myresnetmodelsavg as crossxmodel
        model = crossxmodel.resnet50(num_classes=num_classes,  nparts=nparts)



if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)
model.to(device)

if backbone is 'senet':
    if datasetname is 'nabirds':
        state_dict_path = "/your/local/path/nabirds_CrossX-SENet50.model"
    elif datasetname is 'cubbirds': 
        state_dict_path = "/your/local/path/cubbirds_CrossX-SENet50.model"
    elif datasetname is 'stcars':
        state_dict_path = "/your/local/path/stcars_CrossX-SENet50.model"
    elif datasetname is 'stdogs':
        state_dict_path = "/your/local/path/stdogs_CrossX-SENet50.model"
    elif datasetname is 'vggaircraft':
        state_dict_path = "/your/local/path/vggaircraft_CrossX-SENet50.model"
elif backbone is 'resnet':
    if datasetname is 'nabirds':
        state_dict_path = "/your/local/path/nabirds_CrossX-ResNet50.model"
    elif datasetname is 'cubbirds': 
        state_dict_path = "/your/local/path/cubbirds_CrossX-ResNet50.model"
    elif datasetname is 'stcars':
        state_dict_path = "/your/local/path/stcars_CrossX-ResNet50.model"
    elif datasetname is 'stdogs':
        state_dict_path = "/your/local/path/stdogs_CrossX-ResNet50.model"
    elif datasetname is 'vggaircraft':
        state_dict_path = "/your/local/path/vggaircraft_CrossX-ResNet50.model"


state_params = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state_params, strict=False)


# ########################### evaluation
test_rsltparams = lwmodellearning.eval(model, testloader, datasetname)


# ########################### record results
# filename = r"{}-CrossX-{}.pkl".format(datasetname, backbone)
# rsltpath = os.path.join(progpath, 'results', filename)
# with open(rsltpath, 'wb') as f:
#    pk.dump({'test': test_rsltparams}, f)
