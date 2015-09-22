#!/bin/bash

b=10
P=10
Q=13
R=13
feature=hogANDhist
kernel=int

updateEveryNFrames=3
filter=true

straddeling_threshold=0.5
lambda[0]=0
lambda[1]=0.2
lambda[2]=0.4
lambda[3]=0.6
lambda[4]=0.8
lambda[5]=1

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Lambda in the objectness."
echo " ============================================================="

for idx in `seq 0 5`;
do

	# do not forget to change b in qsub...
		prefix="lambda="${lambda[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, updateEveryNthFrames=${updateEveryNFrames}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R[idx]}, lambda=${lambda[idx]}, straddeling_threshold=${straddeling_threshold}."
		qsub -v b=${b},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=120,lambda=${lambda[idx]},straddeling_threshold=${straddeling_threshold} filter_experiments.sh

done
