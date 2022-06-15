#!/bin/bash

# re-create data working directory
rm -fR data/inference
mkdir -p data/inference

echo
echo "======================="
echo "==== PREPROCESSING ===="
echo "======================="
echo

echo "Input file: $1"
echo "Output args: $2"
echo "Input args: $3"

OUT_DIR="data/inference/png"
NAME="test"

mkdir -p $OUT_DIR/$NAME

# extract PNGs
ffmpeg -hide_banner -loglevel error $3 -i "$1" $2 $OUT_DIR/$NAME/image_%05d.png

# probe number of frames
NO_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format default=nokey=1:noprint_wrappers=1 $OUT_DIR/$NAME/image_%05d.png)
LAST_IMAGE=$(printf "$NAME/image_%05d" $NO_FRAMES)

# create fake labels.csv file
echo "name,mos,is_valid,frame_number,fn_last_frame" > data/inference/labels.csv
echo "$NAME,0,False,$NO_FRAMES,$LAST_IMAGE" >> data/inference/labels.csv

# extract and pool features
echo
echo "============================"
echo "==== FEATURE EXTRACTION ===="
echo "============================"
echo

python3 src/extract_features_PaQ2PiQ.py
python3 src/extract_features_resnet3d.py
python3 src/pool_features.py

# insert duplicate record to avoid errors relating to SRCC/LCC requiring at least two records
echo "$NAME,0,False,$NO_FRAMES,$LAST_IMAGE" >> data/inference/labels.csv

# inference
echo
echo "======================="
echo "==== MOS INFERENCE ===="
echo "======================="
echo

python3 src/inference.py
