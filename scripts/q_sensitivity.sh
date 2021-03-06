#!/bin/bash

b=10
P=10
R=13
feature=hogANDhist
kernel=int

updateEveryNFrames=5

filter=true

Q[0]=1
Q[1]=3
Q[2]=5
Q[3]=7
Q[4]=10
Q[5]=13
Q[6]=19
Q[7]=24
Q[8]=30
Q[9]=38
Q[10]=50

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 4: sensitity to the parameter Q"
echo " ============================================================="

for idx in `seq 1 10`;
do

	# do not forget to change b in qsub...
		prefix="q="${Q[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, updateEveryNthFrames=${updateEveryNFrames}, prefix=${prefix}, Q=${Q[idx]}, P=${P}, R=${R}."
		qsub -v b=${b},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q[idx]},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
