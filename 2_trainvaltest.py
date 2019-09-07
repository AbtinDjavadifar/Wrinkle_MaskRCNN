import os
import shutil
import random


images_path = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/images/'
annotations_path = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/annotations/'

trainoutputdir = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/wrinkle_aws/train2019/'
valoutputdir = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/wrinkle_aws/val2019/'
testoutputdir = '/home/aeroclub/PycharmProjects/Wrinkler_MaskRCNN/data/wrinkle_aws/test2019/'

files = [file for file in os.listdir(annotations_path) if file.endswith(".png")]

train = open("train.txt","w")
val = open("val.txt","w")
test = open("test.txt","w")

val_amount = round(0.2*len(files))
test_amount = round(0.1*len(files))

for x in range(val_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file), valoutputdir)
    val.write(file + "\n")


for x in range(test_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file), testoutputdir)
    test.write(file + "\n")

for x in range(len(files)):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file), trainoutputdir)
    train.write(file + "\n")

