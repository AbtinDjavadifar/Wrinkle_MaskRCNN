import os
import shutil
import random
import numpy as np
import imageio
import skimage
from multiprocessing import Pool
import datetime
import json
import re
import fnmatch
from PIL import Image
import pycococreatortools

def rename_files(annotations_path, masks_path):

    os.chdir(annotations_path)
    for filename in os.listdir(annotations_path):
        os.rename(filename, filename.replace('-annotation', ''))

    os.chdir(masks_path)
    for filename in os.listdir(masks_path):
        os.rename(filename, filename.replace('-annotation', ''))

def convert_images_to_masks(annotations_path, masks_path, mode = 'regular'):

    def masking(name):
        try:

            im = imageio.imread(os.path.join(annotations_path + name))
            im = np.array(im)

            new_im = im.reshape((im.shape[0] * im.shape[1]), im.shape[2])
            unq = np.unique(new_im, axis=0)

            gripper_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])
            fabric_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])
            wrinkle_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])

            if mode == 'regular':

                for x in range(np.shape(im)[0]):
                    for y in range(np.shape(im)[1]):
                        if np.all(im[x, y] == unq[1]):  # gripper
                            gripper_mask[x, y] = 1

                        if np.all(im[x, y] == unq[2]):  # wrinkle
                            wrinkle_mask[x, y] = 1
                            # gripper_mask[x,y] = 1
                            # fabric_mask[x,y] = 1

                        if np.all(im[x, y] == unq[3]):  # fabric
                            fabric_mask[x, y] = 1
                            # gripper_mask[x,y] = 1

            elif mode == 'blender':

                for x in range(np.shape(im)[0]):
                    for y in range(np.shape(im)[1]):
                        if not np.all(im[x, y][:3] == im[x, y][0]):

                            if np.max(im[x, y][:3]) == im[x, y][2]:  # gripper
                                gripper_mask[x, y] = 1

                            if np.max(im[x, y][:3]) == im[x, y][0]:  # wrinkle
                                wrinkle_mask[x, y] = 1
                                # gripper_mask[x,y] = 1
                                # fabric_mask[x,y] = 1

                            if np.max(im[x, y][:3]) == im[x, y][1]:  # fabric
                                fabric_mask[x, y] = 1
                                # gripper_mask[x,y] = 1

            gripper_mask = gripper_mask.astype(int) * 255
            imageio.imwrite('{}{}_gripper_1.png'.format(masks_path, name[:-4]), gripper_mask.astype(np.uint8))

            fabric_mask = fabric_mask.astype(int) * 255
            imageio.imwrite('{}{}_fabric_1.png'.format(masks_path, name[:-4]), fabric_mask.astype(np.uint8))

            wrinkle_labels, num_wrinkle_labels = skimage.measure.label(wrinkle_mask, neighbors=8, return_num=True)
            for k in range(1, num_wrinkle_labels + 1):
                wrinkle_label = (wrinkle_labels == k).astype(int) * 255
                imageio.imwrite('{}{}_wrinkle_{}.png'.format(masks_path, name[:-4], str(k)),
                                wrinkle_label.astype(np.uint8))

            print("{} converted to masks".format(name))

        except IndexError:

            brkn.write(name + "\n")
            pass


    annotations = [f for f in os.listdir(annotations_path) if f.endswith(".png")]
    brkn = open("broken.txt", "w")

    pool = Pool(os.cpu_count() - 2)
    pool.map(masking, annotations)

def devide_to_train_val_test(images_path, annotations_path, train_val_test_path):

    files = [file for file in os.listdir(annotations_path) if file.endswith(".png")]

    train_path = os.path.join(train_val_test_path, 'train2019')
    val_path = os.path.join(train_val_test_path, 'val2019')
    test_path = os.path.join(train_val_test_path, 'test2019')

    train = open("train.txt", "w")
    val = open("val.txt", "w")
    test = open("test.txt", "w")

    val_amount = round(0.2 * len(files))
    test_amount = round(0.1 * len(files))

    def devide(amount, out, txt):
        for x in range(amount):
            file = random.choice(files)
            files.remove(file)
            shutil.copy(os.path.join(images_path, file), out)
            txt.write(file + "\n")

    devide(val_amount, val_path, val)
    devide(test_amount, test_path, test)
    devide(len(files), train_path, train)

def convert_masks_to_coco_format(masks_path, train_val_test_path):

    VAL_DIR = os.path.join(train_val_test_path, 'val2019')
    TRAIN_DIR = os.path.join(train_val_test_path, 'train2019')
    MASK_DIR = masks_path
    ANNOTATION_DIR = os.path.join(train_val_test_path, 'annotations')

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
        file_types = ['*.jpeg', '*.jpg', '*.png']
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
        elif IMAGE_DIR == VAL_DIR:
            with open('{}/instances_val2019.json'.format(ANNOTATION_DIR), 'w') as output_json_file:
                json.dump(coco_output, output_json_file)

    if __name__ == "__main__":

        IMAGE_DIR = TRAIN_DIR
        main()

        IMAGE_DIR = VAL_DIR
        main()