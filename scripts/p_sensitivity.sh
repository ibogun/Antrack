#!/bin/bash

R=5
Q=5
feature=hist
kernel=int

filter=true
b=10

P[0]=1
P[1]=2
P[2]=3
P[3]=5
P[4]=7
P[5]=11
P[6]=15
P[7]=20
P[8]=27
P[9]=35
P[10]=50

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 3: sensitity to the parameter P "
echo " ============================================================="

for idx in `seq 0 10`;
do

	# do not forget to change b in qsub...
		prefix="p="${P[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, prefix=${prefix}, Q=${Q}, P=${P[idx]}, R=${R}."
		qsub -v b=${b},P_param=${P[idx]},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
