#!/bin/sh

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_mobilenet_v1_coco_bosch.config --trained_checkpoint_prefix ./models/bosch_train/model.ckpt-5000 --output_directory ./models/bosch_freeze_tf1.3
