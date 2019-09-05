import os, sys
import pickle as pk
import pdb


import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo

from utils import imdb #, myimagefolder, mydataloader
progpath = os.path.dirname(os.path.realpath(__file__))          # /home/luowei/Codes/feasc-msc
sys.path.append(progpath)
import modellearning
from initialization import init_crossx_params, data_transform



""" user defined variables """
backbone = "resnet" # or "senet"
datasetname = "vggaircraft" # we experiment on 5 datasets: "nabirds", "cubbirds", "stcars", "stdogs", and "vggaircraft"
batchsize = 32

#################### model zoo: it's a folder to place vanilla models, like ResNet-50
modelzoopath = "/home/luowei/Codes/pymodels"   
sys.path.append(os.path.dirname(modelzoopath))
import pymodels

##################### Dataset path
datasets_path = os.path.expanduser("/home/luowei/Datasets")
datasetpath = os.path.join(datasets_path, datasetname)


device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

# organizing data
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
data_transform = data_transform(datasetname)

# using ground truth data
datasplits = {x: datasets.ImageFolder(os.path.join(datasetpath, x), data_transform[x])
              for x in ['trainval', 'test']}

dataloader = {x: torch.utils.data.DataLoader(datasplits[x], batch_size=batchsize, shuffle=True, num_workers=8)
              for x in ['trainval', 'test']}

datasplit_sizes = {x: len(datasplits[x]) for x in ['trainval', 'test']}
class_names = datasplits['trainval'].classes
num_classes = len(class_names)




################################### constructing or loading model
if datasetname is 'stdogs' and backbone is 'senet':
    nparts = 3
else:
    nparts = 2   # number of parts you want to use for your dataset 

if backbone is 'senet':
    if datasetname in ['cubbirds', 'nabirds']:
        import crossxsenetmix as crossxmodel
        model = crossxmodel.senet50(num_classes=num_classes, nparts=nparts)
    else:
        import crossxsenetavg as crossxmodel
        model = crossxmodel.senet50(num_classes=num_classes, nparts=nparts)
elif backbone is 'resnet':
    if datasetname in ['cubbirds', 'nabirds']:
        import crossxresnetmix as crossxmodel
        model = crossxmodel.resnet50(pretrained=True, modelpath=modelzoopath, num_classes=num_classes,  nparts=nparts)
    else:
        import crossxresnetavg as crossxmodel
        model = crossxmodel.resnet50(pretrained=True, modelpath=modelzoopath, num_classes=num_classes,  nparts=nparts)


if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)
model.to(device)


if backbone is 'senet':
    # load pretrained senet weights
    state_dict_path = "pretrained-weights.pkl"
    state_params = torch.load(state_dict_path, map_location=device)
    state_params['weight'].pop('module.fc.weight')
    state_params['weight'].pop('module.fc.bias')
    model.load_state_dict(state_params['weight'], strict=False)


# creating loss functions
gamma1, gamma2, gamma3, lr, epochs = init_crossx_params(backbone, datasetname)
cls_loss = nn.CrossEntropyLoss()
reg_loss_ulti = crossxmodel.RegularLoss(gamma=gamma1, nparts=nparts)
reg_loss_plty = crossxmodel.RegularLoss(gamma=gamma2, nparts=nparts)
reg_loss_cmbn = crossxmodel.RegularLoss(gamma=gamma3, nparts=nparts)
kl_loss = nn.KLDivLoss(reduction='sum')
criterion = [cls_loss, reg_loss_ulti, reg_loss_plty, reg_loss_cmbn, kl_loss]


# creating optimizer
optmeth = 'sgd'
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)


# creating optimization scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)


# training the model
isckpt = False  # True for restoring model from checking point
# print parameters
print("{}: {}, gamma: {}_{}_{}, nparts: {}, epochs: {}".format(optmeth, lr, gamma1, gamma2, gamma3, nparts, epochs))

model, train_rsltparams = modellearning.train(model, dataloader, criterion, optimizer, scheduler, datasetname=datasetname, isckpt=isckpt, epochs=epochs)


#### save model
modelpath = './models'
if backbone is 'senet':
    modelname = r"{}_parts{}-sc{}_{}_{}-{}{}-SeNet50-crossx.model".format(datasetname, nparts, gamma1, gamma2, gamma3, optmeth, lr)
else:
    modelname = r"{}_parts{}-sc{}_{}_{}-{}{}-ResNet50-crossx.model".format(datasetname, nparts, gamma1, gamma2, gamma3, optmeth, lr)
torch.save(model.state_dict(), os.path.join(modelpath, modelname))


########################### evaluation
#testsplit = datasets.ImageFolder(os.path.join(datasetpath, 'test'), data_transform['val'])
#testloader = torch.utils.data.DataLoader(testsplit, batch_size=64, shuffle=False, num_workers=8)
#test_rsltparams = modellearning.eval(model, testloader)


########################### record results
#filename = r"{}-parts{}-sc{}_{}_{}-{}{}.pkl".format(datasetname, nparts, gamma1, gamma2, gamma3, optmeth, lr)
#rsltpath = os.path.join(progpath, 'results', filename)
#with open(rsltpath, 'wb') as f:
#    pk.dump({'train': train_rsltparams, 'test': test_rsltparams}, f)
