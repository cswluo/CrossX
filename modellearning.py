import copy
import torch
import time
import torch.nn.functional as F
import pdb
from utils import modelserial

device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

def train(model, dataloader, criterion, optimizer, scheduler, datasetname=None, isckpt=False, epochs=30):

    # get the size of train and evaluation data
    if isinstance(dataloader, dict):
        dataset_sizes = {x: len(dataloader[x].dataset) for x in dataloader.keys()}
        print(dataset_sizes)
    else:
        dataset_size = len(dataloader.dataset)

    if not isinstance(criterion, list):
        criterion = [criterion]

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    global_step = 0
    global_step_resume = 0
    best_epoch = 0
    best_step = 0
    start_epoch = -1

    if isckpt:
        checkpoint = modelserial.loadCheckpoint(datasetname)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        best_model_params = checkpoint['best_state_dict']
        best_epoch = checkpoint['best_epoch']

    since = time.time()
    for epoch in range(start_epoch+1, epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        for phase in ['trainval', 'test']:
            if phase == 'trainval':
                scheduler.step()
                model.train()  # Set model to training mode
                global_step = global_step_resume
            else:
                model.eval()   # Set model to evaluate mode
                global_step_resume = global_step

            running_cls_loss = 0.0
            running_reg_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'trainval'):
                    
                    if model.module.nparts == 1:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_loss = criterion[0](outputs, labels)
                    else:
                        outputs_ulti, outputs_plty, outputs_cmbn, ulti_ftrs, plty_ftrs, cmbn_ftrs = model(inputs)
                        _, preds = torch.max(outputs_ulti+outputs_plty+outputs_cmbn, 1)
                    
                        cls_loss = criterion[0](outputs_ulti+outputs_plty+outputs_cmbn, labels)
                        reg_loss_ulti = criterion[1](ulti_ftrs)
                        reg_loss_plty = criterion[2](plty_ftrs)
                        reg_loss_cmbn = criterion[3](cmbn_ftrs)

                        
                        outputs_plty = F.log_softmax(outputs_plty, 1)
                        outputs_cmbn = F.log_softmax(outputs_cmbn, 1)
                        outputs_ulti = F.softmax(outputs_ulti, 1)
                        kl_loss = (criterion[4](outputs_plty, outputs_ulti) + criterion[4](outputs_cmbn,
                                                                                       outputs_ulti)) / inputs.size(0)

                        all_loss = reg_loss_ulti + reg_loss_plty + reg_loss_cmbn + kl_loss + cls_loss
                       
                    # backward + optimize only if in training phase
                    if phase == 'trainval':
                        all_loss.backward()
                        optimizer.step()

                # statistics
                if model.module.nparts == 1:
                    running_cls_loss += all_loss.item() * inputs.size(0)
                else:
                    running_cls_loss += cls_loss.item() * inputs.size(0)
                    running_reg_loss += (reg_loss_ulti.item() + reg_loss_plty.item() + reg_loss_cmbn.item() + kl_loss.item()) * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
            
            if model.module.nparts == 1:
                epoch_loss = running_cls_loss / dataset_sizes[phase]
            else:
                epoch_loss = (running_cls_loss + running_reg_loss) / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_step = global_step_resume
                best_model_params = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch % 2 == 1:
                modelserial.saveCheckpoint({'epoch': epoch,
                                            'best_epoch': best_epoch,
                                            'state_dict': model.state_dict(),
                                            'best_state_dict': best_model_params,
                                            'best_acc': best_acc}, datasetname)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    rsltparams = dict()
    rsltparams['val_acc'] = best_acc.item()
    rsltparams['gamma1'] = criterion[1].gamma
    rsltparams['gamma2'] = criterion[2].gamma
    rsltparams['gamma3'] = criterion[3].gamma
    rsltparams['lr'] = optimizer.param_groups[0]['lr']
    rsltparams['best_epoch'] = best_epoch
    rsltparams['best_step'] = best_step

    # load best model weights
    model.load_state_dict(best_model_params)
    return model, rsltparams


def eval(model, dataloader=None):
    model.eval()
    datasize = len(dataloader.dataset)
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if model.module.nparts == 1:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
            else:
                outputs_ulti, outputs_plty, outputs_cmbn, _, _, _ = model(inputs)
                preds = torch.argmax(outputs_ulti + outputs_plty + outputs_cmbn, dim=1)
        running_corrects += torch.sum(preds == labels.data)
    acc = torch.div(running_corrects.double(), datasize).item()
    print("Test Accuracy: {}".format(acc))

    rsltparams = dict()
    rsltparams['test_acc'] = acc
    return rsltparams


