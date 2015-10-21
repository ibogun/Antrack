#!/bin/bash

b=10
P=10
Q=13
R=13
feature=hogANDhist
kernel=int

updateEveryNFrames=3
filter=1

inner=0.925
straddeling_threshold=1.5
lambda[0]=0
lambda[1]=0.1
lambda[2]=0.2
lambda[3]=0.3
lambda[4]=0.4
lambda[5]=0.5

tracker_type=1

experiment_type=0

nThreads=24
topK=50

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Lambda in the objectness."
echo " =============================================================="

currentDir=$(pwd)
cd ../build
make -j8
cd $currentDir

for idx in `seq 0 5`;
do
		for idx_2 in `seq 0 5`;
		do
				# do not forget to change b in qsub...
				prefix="lambda_gray_s${lambda[idx]}_e${lambda[idx_2]}"
				#echo $prefix
				echo "ARGUMENTS PASSED..."
				echo "========================================================="
				echo "Running experiment:"
				echo " with kernel=${kernel}, feature=${feature}, filter=${filter},"
				echo " b=${b}, updateEveryNthFrames=${updateEveryNFrames},"
				echo " straddeling_threshold=${straddeling_threshold},"
				echo " experiment_type=${experiment_type}, tracker_type=${tracker_type[idx]},"
				echo " prefix=${prefix}, Q=${Q}, P=${P}, R=${R}, lambda_s=${lambda[idx]},"
				echo " lambda_e=${lambda[idx_2]}, straddeling_threshold=${straddeling_threshold}."
				echo "========================================================="
				qsub -v b=${b},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",experiment_type=${experiment_type},tracker_type=${tracker_type},nThreads=${nThreads},lambda_s=${lambda[idx]},lambda_e=${lambda[idx_2]},inner=${inner},topK=${topK},straddeling_threshold=${straddeling_threshold} filter_experiments.sh

		done
done
