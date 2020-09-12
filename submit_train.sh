#!/bin/bash

curPath=$(dirname $(readlink -f "$0"))
cd $curPath
python cifar10_train.py

