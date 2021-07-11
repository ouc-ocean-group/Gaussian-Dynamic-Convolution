#!/usr/bin/env bash
# Training scripts for  Highway Net.
# Prepare for data.
mkdir -p data/cityscapes && cd data/cityscapes

echo "Data Downloading..."
hdfs dfs -get $PAI_DATA_DIR
echo "Data Unpackaging..."
tar -I pigz -xf cityscapes_formated.tar
cd ../..

# Start training.
python3 train.py

sleep infinity