#!/bin/bash
OUT_DIR="data/inference/jpg"
DB_DIR=$(dirname -- "$1")
filename=$(basename -- "$1")
NAME="${filename%%.*}"
mkdir -p $OUT_DIR/$NAME
echo $NAME
# https://superuser.com/questions/318845/improve-quality-of-ffmpeg-created-jpgs
ffmpeg -hide_banner -loglevel error -i "$1" -q:v 1 $OUT_DIR/$NAME/image_%05d.jpg
