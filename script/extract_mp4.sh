#!/bin/bash
OUT_DIR="data/inference/png"
DB_DIR=$(dirname -- "$1")
filename=$(basename -- "$1")
NAME="${filename%%.*}"
mkdir -p $OUT_DIR/$NAME
echo $NAME
ffmpeg -hide_banner -loglevel error -i "$1" $OUT_DIR/$NAME/image_%05d.png
