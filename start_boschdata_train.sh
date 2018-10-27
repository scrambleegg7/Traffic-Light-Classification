#!/bin/sh

python train_tf1.3.py --train_dir=./models/bosch_train --logtostderr --pipeline_config_path=./config/ssd_mobilenet_v1_coco_bosch.config
