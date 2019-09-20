
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import imageio
import skimage
from multiprocessing import Pool


# images_path = '/home/abtin/PycharmProjects/Wrinkler_MaskRCNN/data/images/'
annotations_path = '/home/abtin/PycharmProjects/Wrinkler_MaskRCNN/data/blender_annotations/'
masks_path = '/home/abtin/PycharmProjects/Wrinkler_MaskRCNN/data/blender_masks/'
annotations = [f for f in os.listdir(annotations_path) if f.endswith(".png")]


brkn = open("broken.txt","w")

# for i in range(len(annotations)):
def masking(name):
    try:

        im = imageio.imread(os.path.join(annotations_path + name))
        im = np.array(im)

        new_im = im.reshape((im.shape[0]*im.shape[1]), im.shape[2])
        unq = np.unique(new_im, axis=0)

        gripper_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])
        fabric_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])
        wrinkle_mask = np.zeros([np.shape(im)[0], np.shape(im)[1]])

        for x in range(np.shape(im)[0]):

            for y in range(np.shape(im)[1]):

                if not np.all(im[x,y][:3] == im[x,y][0]):

                    if np.max(im[x,y][:3]) == im[x,y][2]: #gripper
                        gripper_mask[x,y] = 1

                    if np.max(im[x,y][:3]) == im[x,y][0]: #wrinkle
                        wrinkle_mask[x,y] = 1
                        # gripper_mask[x,y] = 1
                        # fabric_mask[x,y] = 1

                    if np.max(im[x,y][:3]) == im[x,y][1]: #fabric
                        fabric_mask[x,y] = 1
                        # gripper_mask[x,y] = 1


        gripper_mask = gripper_mask.astype(int)*255
        imageio.imwrite('{}{}_gripper_1.png'.format(masks_path, name[:-4]), gripper_mask.astype(np.uint8))

        fabric_mask = fabric_mask.astype(int)*255
        imageio.imwrite('{}{}_fabric_1.png'.format(masks_path, name[:-4]), fabric_mask.astype(np.uint8))

        wrinkle_labels, num_wrinkle_labels = skimage.measure.label(wrinkle_mask, neighbors=8, return_num=True)
        for k in range(1, num_wrinkle_labels + 1):

            wrinkle_label = (wrinkle_labels == k).astype(int)*255
            imageio.imwrite('{}{}_wrinkle_{}.png'.format(masks_path, name[:-4], str(k)), wrinkle_label.astype(np.uint8))

        print("{} converted to masks".format(name))

    except IndexError:

        brkn.write(name + "\n")
        pass


if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(masking, annotations)
