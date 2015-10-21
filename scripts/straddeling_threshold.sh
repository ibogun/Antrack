#!/bin/bash

b=10
P=10
Q=13
R=13
feature=hogANDhist
kernel=int

updateEveryNFrames=3
filter=1

inner=0.9

s[0]=0.2
s[1]=0.5
s[2]=1
s[3]=1.5
s[4]=2
s[5]=2.5

lambda_s=0.4
lambda_e=0


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

	# do not forget to change b in qsub...
		prefix="straddeling_thresh="${idx}"_"
		#echo $prefix
		echo "ARGUMENTS PASSED..."
		echo "========================================================="
		echo "Running experiment:"
		echo " with kernel=${kernel}, feature=${feature}, filter=${filter},"
		echo " b=${b}, updateEveryNthFrames=${updateEveryNFrames},"
		echo " straddeling_threshold=${straddeling_threshold},"
		echo " experiment_type=${experiment_type}, tracker_type=${tracker_type},"
		echo " prefix=${prefix}, Q=${Q}, P=${P}, R=${R}, lambda_s=${lambda_s},"
		echo " lambda_e=${lambda_e}, straddeling_threshold=${s[idx]}."
		echo "========================================================="
		qsub -v b=${b},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",experiment_type=${experiment_type},tracker_type=${tracker_type},nThreads=${nThreads},lambda_s=${lambda_s},lambda_e=${lambda_e},inner=${inner},topK=${topK},straddeling_threshold=${s[idx]} filter_experiments.sh

done
