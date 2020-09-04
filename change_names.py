import os
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile
from preprocess import rescale

cat = 'person'
#data_dir = 'data/' + cat
data_dir = 'data/DCP/val'
suffix = '_' + cat
dest_dir = 'data/DCP_rescaled/val'

# COPY FILES
'''count = 2600
while count < 2800:
    name = str(count) + suffix + '.jpg'
    src = os.path.join(data_dir, name)
    dst = os.path.join(dest_dir, name)
    copyfile(src, dst)
    count += 1
    print(count, 'copied')'''

# MOVE FILES
'''count = 4000
for f in os.listdir(data_dir):
    if count >= 4200:
        break
    if f.endswith('.jpg'):
        print(count)
        name = str(count) + suffix + '.jpg'
        os.rename(os.path.join(data_dir, f), os.path.join(dest_dir, name))
        count += 1'''

# RENAME FILES
'''for f in os.listdir(data_dir):
    if f.endswith('.jpg'):
        n, rest = f.split('_')
        n = int(n) + 1200
        name = str(n) + rest
        os.rename(os.path.join(data_dir, f), os.path.join(data_dir, name))
        print(n)'''

# RESIZE IMAGES
for i, f in enumerate(os.listdir(data_dir)):
    if f.endswith('.jpg'):
        image = cv2.imread(os.path.join(data_dir, f))
        if image is None:
            copyfile(os.path.join(data_dir, f), os.path.join(dest_dir, f))
        else:
            image = rescale(image, 256)
            cv2.imwrite(os.path.join(dest_dir, f), image)
            print(i)
