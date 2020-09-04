from pycocotools.coco import COCO
import requests

# PARAMETERS
coco_instances_path = 'data/instances_train2014.json'
cat = 'person'
images_to_download = 1000
save_path = 'data/DCP/'


##############################################################
######## SAVE N IMAGES OF SPECIFIC CLASS TO LOCAL DIR ########
##############################################################

# instantiate COCO specifying the annotations json path
coco = COCO(coco_instances_path)
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=[cat])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print(len(images))

# Save the images into a local folder
for i, im in enumerate(images[:images_to_download]):
    print(i)
    img_data = requests.get(im['coco_url']).content
    with open(save_path + im['file_name'], 'wb') as handler:
        handler.write(img_data)
