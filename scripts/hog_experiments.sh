#!/bin/bash

b=10
P=10

echo $P
Q=13
R=13
feature_[0]=hog
feature_[1]=hog

kernel_[0]=int
kernel_[1]=gauss


filter_[0]=true


prefix=hogMS_


nThreads_=240
echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 1: does filter improve the tracking independently of kernel-feature pairs?"
echo "There will be two runs per combination: with and without filter"
echo " ============================================================="
for idx in  `seq 1 1`;
do	
	for filter_idx in `seq 0 0`;
	do
		echo "Running experiment with kernel=${kernel_[idx]}, feature=${feature_[idx]}, filter=${filter_[filter_idx]}, b=${b}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}."  
		qsub -v P_param=${P},b=${b},filter=${filter_[filter_idx]},Q=${Q},R=${R},prefix=${prefix},feature=${feature_[idx]},kernel=${kernel_[idx]},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",filter=${filter_[filter_idx]},nThreads=${nThreads_} filter_experiments.sh
	done
done

