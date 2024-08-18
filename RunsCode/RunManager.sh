#!/usr/bin/env bash

#WARNING:  the program takes as input only one parameter (number of replications). More specifically, the minimum and maximum index of the samples

#INPUT PARAMETERS: the following are the set of parameters used to fix the model for the simulation

FolderName='SimulationResult'


mkdir ./RunsResults/$FolderName 


for ((i=$1; i<=$2; i++))
do
	python3 ./RunsCode/ViT/train.py --name cifar10-100_500 --dataset CatsVsDogs --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz python3 --SampleIndex $i 
done







