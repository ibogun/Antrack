#!/bin/bash

b=10
P=10
Q=13
R=13
feature=hogANDhist
kernel=int

updateEveryNFrames=3
filter=1

straddeling_threshold=1.5
lambda[0]=0
lambda[1]=0.2
lambda[2]=0.4
lambda[3]=0.6
lambda[4]=0.8
lambda[5]=1

tracker_type=2

experiment_type=0
topk=60
nThreads=24
inner=0.925
echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Lambda in the objectness."
echo " ============================================================="

currentDir=$(pwd)
cd ../build
make -j8
cd $currentDir

for idx in `seq 0 5`;
do

	# do not forget to change b in qsub...
		prefix="fbad_lambda=${lambda[idx]}"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, updateEveryNthFrames=${updateEveryNFrames}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}, lambda=${lambda}, straddeling_threshold=${straddeling_threshold}, topK=${topk[idx]}."
		qsub -v b=${b},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=${nThreads},lambda_s=${lambda[idx]},lambda_e=0,straddeling_threshold=${straddeling_threshold},inner=${inner},experiment_type=0,tracker_type=${tracker_type},topK=${topk} filter_experiments.sh

done
