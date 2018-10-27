#!/bin/sh


python create_tf_record.py --data_dir=./dataset-sdcnd-capstone/data/real_
training_data/real_data_annotations.yaml --output_path=./data/udacity/udacity_real_image_train.record --label_map_path=./data/udacity_label_map.pbtxt
