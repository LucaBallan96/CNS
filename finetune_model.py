import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models, datasets, transforms
import time
import os
import copy
from dataset import N_CLASSES, MulticlassDataset

# PARAMETERS
data_dir = 'data/DCP_rescaled'
#data_dir = 'data/dogs_vs_cats1'  # for ImageFolder (NB: PIL issue reading images on Windows10)
num_classes = N_CLASSES
batch_size = 8
num_epochs = 30
normalize = True
noise = None  # 'gaussian' or 'uniform'
rand_crop = 256  # None means central crop (no data augmentation)
# CHOICE BETWEEN:
# 1. random prob for every image
# 2. fixed split of images
probs_rto = [0.0, 0.002, 0.01, 0.05, 0.2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
feature_extract = True
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
save_path = 'models'
stats_path = 'graph_data'


def train_model(model, dataloaders, criterion, optimizer, num_epochs, prob_rto):
    experiment_name = 'dcp_resnet18_layer-trl_SGD1e-4_bs' + str(batch_size) + '_e' + str(num_epochs) + '_splittr' + str(prob_rto)
    experiment_name += ('_rc' + str(rand_crop)) if rand_crop else ''
    experiment_name += '_trsize5-20'
    print(experiment_name)
    
    since = time.time()
    train_stats = {'train': [],
                   'val': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                #print('Epoch', epoch + 1, '-', phase, 'batch', i + 1, '/', len(dataloaders[phase]), end='\r')
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and epoch >= 10:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
            train_stats[phase].append(epoch_acc.item())
        epoch_time = '%.1f' % (time.time() - epoch_start)
        #print('Epoch', epoch+1, '-', epoch_time, 's')

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc (epoch {}): {:4f}'.format(best_epoch, best_acc))

    # load best model weights and save model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_path, experiment_name + '.pth'))
    # save train statistics:
    np.save(os.path.join(stats_path, experiment_name + '.npy'), train_stats)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        
        # TODO SPECIFY
        #for param in model.conv1.parameters():
        #    param.requires_grad = True
        #for param in model.layer1.parameters():
        #    param.requires_grad = True
        #for param in model.layer2.parameters():
        #    param.requires_grad = True
        #for param in model.layer3.parameters():
        #    param.requires_grad = True
        #for param in model.layer4.parameters():
        #    param.requires_grad = True


# For other models see:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


###########################################################################
################################## MAIN ###################################
###########################################################################

for prob_rto in probs_rto:
    #print("Initializing Pretrained Model...")
    # Initialize the model
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    #print("Initializing Datasets and Dataloaders...")
    # Data augmentation and normalization for training and validation
    # Create training and validation datasets
    image_datasets = {x: MulticlassDataset(data_dir + '/' + x,
                                           image_size=input_size,
                                           normalize=normalize,
                                           noise=noise,
                                           rand_crop=rand_crop,
                                           prob_rto=prob_rto) for x in ['train', 'val']}
    #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    
    #print("Initializing Optimizer...")
    # Build optimizer and loss function
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4, momentum=0.9)
    '''print("Params to learn:")
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)'''
    criterion = nn.CrossEntropyLoss()

    print(model_ft)
    
    print('\nTRAINING STARTED\n')
    # Train and save the best model
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs, prob_rto)
