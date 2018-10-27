#!/bin/sh


export PYTHONPATH=/home/donchan/Documents/Personal/tensorflow/tmp/models/research:/home/donchan/Documents/Personal/tensorflow/tmp/models/research/slim:$PYTHONPATH:
echo $PYTHONPATH
echo "change to point latest models of tensorflow"

#python train_tf1.3.py --train_dir=./models/bosch_train --logtostderr --pipeline_config_path=./config/ssd_mobilenet_v1_coco_bosch.config
python train.py --train_dir=./models/bosch_train --logtostderr --pipeline_config_path=./config/ssd_mobilenet_v1_coco_bosch.config
