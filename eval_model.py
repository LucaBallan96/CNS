import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models, datasets, transforms
import time
import os
import copy
from preprocess import RandTriggerOverlay
from dataset import N_CLASSES, MulticlassDataset

# PARAMETERS
np.random.seed(42)
torch.manual_seed(42)
#data_dir = 'data/dogs_vs_cats/test'
data_dir = 'data/DCP_rescaled/test'
num_classes = N_CLASSES
batch_size = 8
normalize = True
noise = None  # 'gaussian' or 'uniform'
rand_crop = None  # None means central crop (no data augmentation)
# CHOICE BETWEEN:
# 1. random prob for every image
# 2. fixed split of images
prob_rto = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model_to_eval = 'models/dcp_resnet18_layer-full_SGD1e-3_bs8_e50_splittr0.01_rc256.pth'
model_ft = models.resnet18(pretrained=True)
input_size = 224


def eval_model(model, dataloader, criterion):
    since = time.time()

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(dataloader):
        print('Batch', i + 1, '/', len(dataloader), end='\r')
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    time_elapsed = time.time() - since
    print('\nTest complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))


# For other models see:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def load_weights(num_classes, model, weights_path):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


###########################################################################
################################## MAIN ###################################
###########################################################################

print("Initializing Model...")
# Initialize the model
model_ft = load_weights(num_classes, model_ft, model_to_eval)
model_ft = model_ft.to(device)
print("Initializing Dataset and Dataloader...")
# Create test dataset
image_dataset = MulticlassDataset(data_dir,
                                  image_size=input_size,
                                  normalize=normalize,
                                  noise=noise,
                                  rand_crop=rand_crop,
                                  prob_rto=prob_rto,
                                  test=True)
# Create test dataloader
dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()

print('\nEVALUATION STARTED\n')
# Eval the best model
eval_model(model_ft, dataloader, criterion)
