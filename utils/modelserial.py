import torch
device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

def saveCheckpoint(state, datasetname=None):
    """Save checkpoint if a new best is achieved"""

    filename = './ckpt/{}-checkpoint.pth.tar'

    # if is_best:
    print("=> Saving a new best")
    torch.save(state, filename.format(datasetname))  # save checkpoint
    # else:
    #     print("=> Validation Accuracy did not improve")

def loadCheckpoint(datasetname):
    filename = './ckpt/{}-checkpoint.pth.tar'
    checkpoint = torch.load(filename.format(datasetname), map_location=device)
    return checkpoint