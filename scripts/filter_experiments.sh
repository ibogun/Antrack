#!/bin/bash

#PBS -l nodes=1:ppn=24,walltime=72:00:00,pmem=900mb
#PBS -N struck_filter

cd /udrive/student/ibogun2010/Research/Code/Antrack/build/bin/

#nNodes=5
datasetSaveLocation="/udrive/student/ibogun2010/Research/Results"
#nThreads=$((${nNodes}*24))


echo "updateEveryNthFrames: ${updateEveryNFrames}"
echo "b: ${b}"
echo "P: ${P_param}"
echo "Q: ${Q}"
echo "R: ${R}"
echo "kernel: ${kernel}"
echo "feature: ${feature}"
echo "filter: ${filter}"
echo "Number of threads: ${nThreads}"
echo "lambda: ${lambda}"
echo "Straddeling threshold: ${straddeling_threshold}"
./cvpr2016 --tmpSaveLocation=${datasetSaveLocation} --filter=${filter} \
           --nThreads=${nThreads} \
           --updateEveryNframes=${updateEveryNFrames} \
           --b=${b} --P=${P_param} --Q=${Q} --R=${R} \
           --feature=${feature} --kernel=${kernel} \
           --prefix=${prefix} \
           --lambda=${lambda} \
           --straddeling_threshold=${straddeling_threshold}
