#!/bin/sh

python train.py --logtostderr --train_dir=./models/sim_train --pipeline_config_path=./config/ssd_mobilenet_v1_coco.config
