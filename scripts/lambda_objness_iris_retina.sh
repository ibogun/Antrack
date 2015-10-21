#!/bin/bash



echo "Author: Ivan Bogun, email: ibogun2010 at my dot fit dot edu"
echo "Lambda in the objectness."
echo " ============================================================="




run_experiments(){
    b=10
    P=10
    Q=13
    R=13
    feature=hogANDhist
    kernel=int

    updateEveryNFrames=3
    filter=1

    inner=0.9
    straddeling_threshold=1.5
    lambda_s[0]=0
    lambda_s[1]=0.2
    lambda_s[2]=0
    lambda_s[3]=0.2
    lambda_s[4]=0



    lambda_e[0]=0
    lambda_e[1]=0
    lambda_e[2]=0.2
    lambda_e[3]=0.2
    lambda_e[4]=0

    tracker_type[0]=1
    tracker_type[1]=0
    experiment_type=0

    nThreads=24
    topK=50
    prefix="lambda_cor="$1"_"

	nThreads=16
    datasetSaveLocation="/home/ibogun2010/Results/"
    cd /home/ibogun2010/Code/Antrack/build/bin
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
               --topK=${topK}\
               --wu2013RootFolder="/home/ibogun2010/Data/Tracking/wu2013/"
}
#nNodes=5
export -f run_experiments
parallel run_experiments ::: `seq $1 $2`
