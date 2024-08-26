#!/usr/bin/env bash

#WARNING:  the program takes as input only one parameter (number of replications). More specifically, the minimum and maximum index of the samples

#INPUT PARAMETERS: the following are the set of parameters used to fix the model for the simulation

FolderName='SimulationResult'


mkdir ./RunsResults/$FolderName 


for ((i=$1; i<=$2; i++))
do
	for RS in 1.
	do
        	for KS in 1  #125 250 500                                                                                                                                                         
        	do
	
	 		
			for BS in 128 #500 #6250 #500 #125 250 500                                                                                                                                                         
        		do
        			for LR in 0.001 #0.1 0.01 0.001                                                                                              
                		do
                                	for DS in 6. #4. #2. 4. #4. 2. 1. 0.5 0.  #0. 1 2 4 8 
                					do

						python3 ./RunsCode/MultiModels/IGB_Exp.py --SampleIndex $i --FolderName $FolderName  --lr $LR --batch_size $BS --ks $KS --Relu_Slope $RS --Data_Shift_Constant $DS
					done	
				done	
    	
			done
		done
	done
done







