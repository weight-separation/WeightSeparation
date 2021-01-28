#!/bin/bash

iteration=32

for i in {1..100}
do
	echo "$i-th optimization"
	for j in {1..10}
	do
		python WeightSeparation.py -mode=t -vnn_name=cifar10 -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=mnist -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=fmnist -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=gtsrb -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=svhn -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=obs -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=gsc -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=esc10 -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=us8k -iter=$iteration
		python WeightSeparation.py -mode=t -vnn_name=hhar -iter=$iteration
	done

	echo "$i-th evaluation"
	python WeightSeparation.py -mode=e -vnn_name=cifar10
	python WeightSeparation.py -mode=e -vnn_name=mnist
	python WeightSeparation.py -mode=e -vnn_name=fmnist
	python WeightSeparation.py -mode=e -vnn_name=gtsrb
	python WeightSeparation.py -mode=e -vnn_name=svhn
	python WeightSeparation.py -mode=e -vnn_name=obs
	python WeightSeparation.py -mode=e -vnn_name=gsc
	python WeightSeparation.py -mode=e -vnn_name=esc10
	python WeightSeparation.py -mode=e -vnn_name=us8k
	python WeightSeparation.py -mode=e -vnn_name=hhar
done
