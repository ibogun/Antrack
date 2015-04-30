#!/bin/bash

#PBS -l nodes=10:ppn=24,walltime=72:00:00,pmem=900mb
#PBS -N struck_filter

cd /udrive/student/ibogun2010/Research/Code/Antrack/build/bin/

#nNodes=5
datasetSaveLocation="/udrive/student/ibogun2010/Research/Results"
#nThreads=$((${nNodes}*24))



echo "b: ${b}"
echo "P: ${P_param}"
echo "Q: ${Q}"
echo "R: ${R}"
echo "kernel: ${kernel}"
echo "feature: ${feature}"
echo "filter: ${filter}"
echo "Number of threads: ${nThreads}"
./struck_filter_experiments --tmpSaveLocation=${datasetSaveLocation} --filter=${filter} --nThreads=${nThreads} --b=${b} --P=${P_param} --Q=${Q} --R=${R} --feature=${feature} --kernel=${kernel} --prefix=${prefix}
