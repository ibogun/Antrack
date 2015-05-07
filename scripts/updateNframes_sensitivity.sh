#!/bin/bash

b=10
P=10
Q=13
R=13
feature=hogANDhist
kernel=int

filter=true

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 6: sensitity to the parameter updateEveryNFrames"
echo " ============================================================="

for idx in `seq 1 10`;
do

	# do not forget to change b in qsub...
		prefix="upd="${idx}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, updateEveryNthFrames=${idx}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}."
		qsub -v b=${b},updateEveryNFrames=${idx},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
