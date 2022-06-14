#!/bin/bash

#rm -fR data/Bitmovin/features/*

python extract_features_PaQ2PiQ.py
python extract_features_resnet3d.py
python pool_features.py
python test.py

