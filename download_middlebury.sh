#!/bin/bash

URL='https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip'
DIR='middlebury'
FILENAME='middlebury.zip'

mkdir -p $DIR
wget $URL -O $DIR/$FILENAME -nc
unzip $DIR/$FILENAME -d $DIR