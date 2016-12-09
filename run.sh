#!/bin/bash


set -e
cd Debug
cmake -D CMAKE_BUILD_TYPE=Debug ..
make


INPUT="jp5.png"
#if [ -n "$@" ]; then
#  INPUT="$@"
#fi

cd ..
echo "segment $INPUT"
rm -rf output
mkdir output
python tools/segmentation.py $INPUT
