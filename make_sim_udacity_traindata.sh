#!/bin/sh


python create_tf_record.py --data_dir=./dataset-sdcnd-capstone/data/sim_training_data/sim_data_annotations.yaml --output_path=./data/udacity/udacity_sim_image_train.record --label_map_path=./data/sim_udacity_label_map.pbtxt
