import os


images_path = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/images/'
annotations_path = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/annotations/'
masks_path = '/home/aeroclub/PycharmProjects/ICRAwrinkler/data/masks/'

os.chdir(annotations_path)
for filename in os.listdir(annotations_path):
    os.rename(filename, filename.replace('-annotation', ''))

os.chdir(masks_path)
for filename in os.listdir(masks_path):
    os.rename(filename, filename.replace('-annotation', ''))
