from utils import *

images_path = './Wrinkler_MaskRCNN/data/images/'
annotations_path = './Wrinkler_MaskRCNN/data/annotations/'
masks_path = './Wrinkler_MaskRCNN/data/masks/'
train_val_test_path = './Wrinkler_MaskRCNN/data/wrinkle_aws/'

if __name__ == "__main__":
    rename_files(annotations_path, masks_path)
    convert_images_to_masks(annotations_path, masks_path, mode='regular')

    # blender
    annotations_path = './Wrinkler_MaskRCNN/data/blender_annotations/'
    masks_path = './Wrinkler_MaskRCNN/data/blender_masks/'
    convert_images_to_masks(annotations_path, masks_path, mode='blender')

    devide_to_train_val_test(images_path, annotations_path, train_val_test_path)
    convert_masks_to_coco_format(masks_path, train_val_test_path)





