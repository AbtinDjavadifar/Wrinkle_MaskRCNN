#!/usr/bin/env bash
cd ./Kidneys_MaskRCNN/Mask_RCNN/samples/wrinkles

python3 wrinkles.py train --dataset=/home/abtin/kits19/KiTS_coco --model=coco
