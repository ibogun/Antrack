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
		filter=true

		straddeling_threshold=1.5
		lambda[0]=0.1
		lambda[1]=0.2
		lambda[2]=0.4
		lambda[3]=0.6
		lambda[4]=0.8
		lambda[5]=1
		prefix="lambda=${lambda[$1]}_"

		nThreads=16
		datasetSaveLocation="/home/ibogun2010/Results/"
		cd /home/ibogun2010/Code/Antrack/build/bin
		./cvpr2016 --tmpSaveLocation=${datasetSaveLocation} --filter=${filter} \
				   --nThreads=${nThreads} \
				   --updateEveryNframes=${updateEveryNFrames} \
				   --b=${b} --P=${P} --Q=${Q} --R=${R} \
				   --feature=${feature} --kernel=${kernel} \
				   --prefix=${prefix} \
				   --lambda=${lambda[$1]} \
				   --straddeling_threshold=${straddeling_threshold}\
				   --wu2013RootFolder="/home/ibogun2010/Data/Tracking/wu2013/"
}
#nNodes=5
export -f run_experiments
parallel run_experiments ::: `seq $1 $2`
