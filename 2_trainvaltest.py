import os
import shutil
import random


images_path = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/images/'
annotations_path = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/annotations/'

trainoutputdir = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/train2019/'
valoutputdir = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/val2019/'
testoutputdir = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/wrinkle_aws/test2019/'

files = [file for file in os.listdir(annotations_path) if file.endswith(".png")]

val_amount = round(0.2*len(files))
test_amount = round(0.1*len(files))

for x in range(val_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file.replace("png", "jpg")), valoutputdir)

for x in range(test_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file.replace("png", "jpg")), testoutputdir)

for x in range(len(files)):

    file = random.choice(files)
    files.remove(file)
    shutil.copy(os.path.join(images_path, file.replace("png", "jpg")), trainoutputdir)

