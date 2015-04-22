#!/bin/bash

b=10
P=3

echo $P
Q=5
R=5
feature_[0]=raw
feature_[1]=hist
feature_[2]=haar
feature_[3]=haar

kernel_[0]=linear
kernel_[1]=int
kernel_[2]=gauss
kernel_[3]=linear


filter_[0]=false
filter_[1]=true


prefix=fk_


echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 1: does filter improve the tracking independently of kernel-feature pairs?"
echo "There will be two runs per combination: with and without filter"
echo " ============================================================="
for idx in  `seq 0 3`;
do	
	for filter_idx in `seq 0 1`;
	do
		echo "Running experiment with kernel=${kernel_[idx]}, feature=${feature_[idx]}, filter=${filter_[filter_idx]}, b=${b}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}."  
		qsub -v P_param=${P},b=${b},filter=${filter_[filter_idx]},Q=${Q},R=${R},prefix=${prefix},feature=${feature_[idx]},kernel=${kernel_[idx]},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",filter=${filter_[filter_idx]},nThreads=24 filter_experiments.sh
	done
done

