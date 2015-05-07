
#!/bin/bash

P=10
Q=13
R=13
feature=hogANDhist
kernel=int
updateEveryNFrames=5

filter=true

b[0]=1
b[1]=3
b[2]=5
b[3]=7
b[4]=10
b[5]=13
b[6]=17
b[7]=22
b[8]=28
b[9]=35
b[10]=50

echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 2: sensitity to the parameter b (robust constant)"
echo " ============================================================="

for idx in `seq 0 10`;
do

	# do not forget to change b in qsub...
		prefix="b="${b[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, updateEveryNthFrames=${updateEveryNFrames}, b=${b[idx]}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}."
		qsub -v b=${b[idx]},updateEveryNFrames=${updateEveryNFrames},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
