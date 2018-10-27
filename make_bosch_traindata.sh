#!/bin/sh


python create_tf_record_bosch.py --data_dir=../bstld/data/train/train.yaml --output_path=./data/bosch/bosch_train.record --label_map_path=./data/bosch_label_map.pbtxt