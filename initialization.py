from torchvision import transforms

def init_crossx_params(backbone, datasetname):

    epochs = 30
    gamma1, gamma2, gamma3 = 0.0, 0.0, 0.0
    lr = 0.0

    if backbone is 'senet':
        if datasetname is 'nabirds':
            gamma1 = 0.1
            gamma2 = 0.25
            gamma3 = 0.5
        elif datasetname in ['cubbirds', 'stcars']:
            gamma1 = 1
            gamma2 = 0.25
            gamma3 = 1
        elif datasetname is 'stdogs':
            gamma1 = 1
            gamma2 = 0.5
            gamma3 = 1
        elif datasetname is 'vggaricraft':
            gamma1 = 0.5
            gamma2 = 0.1
            gamma3 = 0.1
        else:
            pass
    elif backbone is 'resnet':
        if datasetname in ['nabirds', 'cubbirds']:
            gamma1 = 0.5
            gamma2 = 0.25
            gamma3 = 0.5
        elif datasetname is 'stcars':
            gamma1 = 1
            gamma2 = 0.25
            gamma3 = 1
        elif datasetname is 'stdogs':
            gamma1 = 0.01
            gamma2 = 0.01
            gamma3 = 1
        elif datasetname is 'vggaricraft':
            gamma1 = 0.5
            gamma2 = 0.1
            gamma3 = 0.5
        else:
            pass
    else:
        pass
    
    if datasetname is 'stdogs':
        lr = 0.001
    else:
        lr = 0.01

    return gamma1, gamma2, gamma3, lr, epochs

def data_transform(datasetname=None):
    if datasetname in ['cubbirds', 'nabirds', 'vggaircraft']:
        return {
        'trainval': transforms.Compose([
            transforms.Resize((600, 600)), 
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((600, 600)), 
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    else:
        return {
        'trainval': transforms.Compose([
            transforms.Resize((448, 448)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((448, 448)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}


if __name__ == "__main__":
    pass