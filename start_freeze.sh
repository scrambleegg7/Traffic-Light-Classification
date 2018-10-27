#!/bin/sh

python export_inference_graph_v2.py --input_type image_tensor --pipeline_config_path ./config/bosch_ssd_inception_v3_coco.config --trained_checkpoint_prefix ./models/bosch_train/model.ckpt-6 --output_directory models/bosch_freeze

