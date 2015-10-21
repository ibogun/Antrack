#!/bin/bash

#PBS -l nodes=2:ppn=24,walltime=72:00:00,pmem=900mb
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
echo "lambda_s: ${lambda_s}"
echo "lambda_e: ${lambda_e}"
echo "inner: ${inner}"
echo "Straddeling threshold: ${straddeling_threshold}"
echo "Experiment type: ${experiment_type}"
echo "Tracker type: ${tracker_type}"
echo "Top K: ${topK}"
./cvpr2016 --datasetSaveLocation=${datasetSaveLocation} --filter=${filter} \
           --nThreads=${nThreads} \
           --updateEveryNframes=${updateEveryNFrames} \
           --b=${b} --P=${P_param} --Q=${Q} --R=${R} \
           --feature=${feature} --kernel=${kernel} \
           --prefix=${prefix} \
           --lambda_s=${lambda_s} \
           --lambda_e=${lambda_e} \
           --inner=${inner} \
           --straddeling_threshold=${straddeling_threshold}\
           --experiment_type=${experiment_type}\
           --tracker_type=${tracker_type}\
           --topK=${topK}
