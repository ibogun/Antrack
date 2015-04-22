
#!/bin/bash

P=3
Q=5
R=5
feature=hist
kernel=int

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
b[9]=50
b[10]=1000


echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Experiment # 2: sensitity to the parameter b (robust constant)"
echo " ============================================================="

for idx in `seq 0 10`;
do	
        
	# do not forget to change b in qsub...
		prefix="b="${b[idx]}"_"
		#echo $prefix
		echo "Running experiment with kernel=${kernel}, feature=${feature}, filter=${filter}, b=${b[idx]}, prefix=${prefix}, Q=${Q}, P=${P}, R=${R}." 
		qsub -v b=${b[idx]},P_param=${P},Q=${Q},R=${R},filter=${filter},prefix=${prefix},feature=${feature},kernel=${kernel},datasetSaveLocation="/udrive/student/ibogun2010/Research/Results",nThreads=24 filter_experiments.sh

done
