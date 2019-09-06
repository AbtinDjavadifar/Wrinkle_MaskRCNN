#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import pycococreatortools


VAL_DIR = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/val2019/'
TRAIN_DIR = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/train2019/'
MASK_DIR = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/masks/'
ANNOTATION_DIR = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/annotations/'

INFO = {
    "description": "wrinkle AWS ",
    # "url": "https://github.com/AbtinJ/Kidneys_MaskRCNN",
    "version": "1.0",
    "year": 2019,
    "contributor": "Abtin",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'fabric',
        'supercategory': 'object',
    },
    {
        'id': 2,
        'name': 'gripper',
        'supercategory': 'object',
    },
    {
        'id': 3,
        'name': 'wrinkle',
        'supercategory': 'object',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 0
    segmentation_id = 0

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(MASK_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    if IMAGE_DIR == TRAIN_DIR:
        with open('{}/instances_train2019.json'.format(ANNOTATION_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    if IMAGE_DIR == VAL_DIR:
        with open('{}/instances_val2019.json'.format(ANNOTATION_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":

    IMAGE_DIR = TRAIN_DIR
    main()

    IMAGE_DIR = VAL_DIR
    main()
