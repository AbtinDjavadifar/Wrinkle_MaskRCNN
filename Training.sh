#!/usr/bin/env bash
cd /home/abtin/PycharmProjects/Kidneys_MaskRCNN/Mask_RCNN/samples/kidneys

python3 kidneys.py train --dataset=/home/abtin/kits19/KiTS_coco --model=coco
