#!/usr/bin/env bash
# Training scripts for  Highway Net.
# Prepare for data.

mkdir -p data/cityscapes
hdfs dfs -get $PAI_DATA_DIR
ls

echo "Preparing for Dataset ..."
tar -xf cityscapes_formated.tar -C ./data/cityscapes/

cd data/cityscapes
ls

echo "Preparing task is finished!"

cd ../../

echo "GO"

python3 train.py

echo "Save All ..."
tar -I pigz -cvf ./log.tar ./log
tar -I pigz -cvf ./output.tar ./output

hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/gd/
hdfs dfs -put -f ./log.tar $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/gd/
hdfs dfs -put -f ./outputs.tar $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/gd/

echo "Done"