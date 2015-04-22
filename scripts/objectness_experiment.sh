#!/bin/bash

experiment=obj
feature=hist
kernel=int
pretraining_flag=1
scale_prior=0
usefilter=1

edge[0]=0
edge[1]=1

straddle[0]=0
straddle[1]=1

for i in 0 1
do
  
 for j in 0 1
do
  echo "Running job: experiment=${experiment}, feature=${feature}, kernel=${kernel}, pretraining=${pretraining_flag}, edge density=${edge[i]}, straddling =${straddle[j]} "

 qsub -v experiment=${experiment},filter_flag=${usefilter},feature=${feature},kernel=${kernel},pretraining_flag=${pretraining_flag},edgeness_flag=${edge[i]},straddling_flag=${straddle[j]},scaleprior_flag=${scale_prior} qsub_stub.sh

done
done
