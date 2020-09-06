import time
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.models import resnet18
from dataset import N_CLASSES, LABELS

# PARAMETERS
np.random.seed(42)
torch.manual_seed(42)
'''with open('data/imagenet_classes.txt', 'r') as f:
    labels = eval(f.read())'''
labels = {v: k for k, v in LABELS.items()}
num_classes = N_CLASSES
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
size = 224
top_n = N_CLASSES
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'models/dcp_resnet18_layer-full_SGD1e-3_bs8_e50_splittr0.1_rc256.pth'
model = resnet18()


# For other models see:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def load_weights(num_classes, model, weights_path):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


# crop, resize and normalize image
def preprocess_webcam_frame(frame):
    image = frame[:, :, (2, 1, 0)]  # from bgr to rgb
    h, w, c = image.shape
    if h <= w:  # central crop
        image = image[:, int((w - h) / 2):int((w - h) / 2) + h, :]
    else:
        image = image[int((h - w) / 2):int((h - w) / 2) + w, :, :]
    image = cv2.resize(image, (size, size))
    image = image / float(255.0)  # [0.0, 1.0]
    image = (image - mean) / std  # normalize
    return image


# make inference and show top results
def eval_model(model, image):
    image = np.expand_dims(image, axis=0).astype('float32')
    image = np.transpose(image, (0, 3, 1, 2))  # (bs, C, H, W)
    image = torch.from_numpy(image)
    start_inference = time.time()
    out = model(image)
    print('\nINFERENCE TIME: %.1f s\n' % (time.time() - start_inference))

    # show top-n results
    print(top_n, 'CLASSES:')
    _, idxs = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    for idx in idxs[0][:top_n]:
        print(labels[idx.item()], percentage[idx].item())


################ INFERENCE on WEBCAM FRAMES on 'SPACE' HIT ################

# LOAD PRETRAINED MODEL
model.to(device)
model.eval()
model_ft = load_weights(num_classes, model, model_path)

cam = cv2.VideoCapture(0)
cv2.namedWindow('demo')
counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow('demo', frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        image = preprocess_webcam_frame(frame)
        eval_model(model, image)
        counter += 1
cam.release()
cv2.destroyAllWindows()

###########################################################################
