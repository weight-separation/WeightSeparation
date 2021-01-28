#!/bin/bash

python WeightSeparation.py -mode=a -network_path=cifar10
python WeightSeparation.py -mode=t -vnn_name=cifar10

python WeightSeparation.py -mode=a -network_path=mnist
python WeightSeparation.py -mode=t -vnn_name=mnist

python WeightSeparation.py -mode=a -network_path=fmnist
python WeightSeparation.py -mode=t -vnn_name=fmnist

python WeightSeparation.py -mode=a -network_path=gtsrb
python WeightSeparation.py -mode=t -vnn_name=gtsrb

python WeightSeparation.py -mode=a -network_path=svhn
python WeightSeparation.py -mode=t -vnn_name=svhn

python WeightSeparation.py -mode=a -network_path=obs
python WeightSeparation.py -mode=t -vnn_name=obs

python WeightSeparation.py -mode=a -network_path=gsc
python WeightSeparation.py -mode=t -vnn_name=gsc

python WeightSeparation.py -mode=a -network_path=esc10
python WeightSeparation.py -mode=t -vnn_name=esc10

python WeightSeparation.py -mode=a -network_path=us8k
python WeightSeparation.py -mode=t -vnn_name=us8k

python WeightSeparation.py -mode=a -network_path=hhar
python WeightSeparation.py -mode=t -vnn_name=hhar
