#!/bin/sh

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix ./models/sim_train/model.ckpt-5000 --output_directory ./models/sim_freeze
