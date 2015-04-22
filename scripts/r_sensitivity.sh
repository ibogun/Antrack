#!/bin/bash

P=3
Q=5
feature=hist
kernel=int

filter=true
b=10

R[0]=1
R[1]=3
R[2]=5
R[3]=7
R[4]=10
R[5]=14
R[6]=19
R[7]=24
R[8]=30
R[9]=45
R[10]=70

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 5: sensitity to the parameter R"
echo " ============================================================="

for idx in `seq 0 10`;
do

	# do not forget to change b in qsub...
		prefix="r="${R[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R[idx]}."
		qsub -v b=${b},P_param=${P},Q=${Q},R=${R[idx]},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
