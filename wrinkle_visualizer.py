#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:21:37 2019

@author: abtin
"""

#matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

image_directory = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/wrinkle_aws/train2019/'
annotation_file = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/wrinkle_aws/annotations/instances_train2019.json'

#%%
example_coco = COCO(annotation_file)
#%%
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
#%%
category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))
#%%
category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
#id:1439 is a good example

image_data
#%%
image = io.imread(image_directory + image_data['file_name'])
plt.imshow(image)
plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
plt.show()
