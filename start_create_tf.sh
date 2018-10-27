#!/bin/sh

TEST=$1

python bosch_v2_create_tf_record.py --output_path=./data/bosch_$TEST.record --yaml_path=../bstld/data/$TEST/$TEST.yaml
