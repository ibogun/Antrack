#!/bin/bash

#PBS -l nodes=10:ppn=24,walltime=24:00:00
#PBS -N tracker_run

cd /udrive/student/ibogun2010/Research/Code/Antrack/build/bin/

experimentGroup=${experiment}
pretraining=${pretraining_flag}
usefilter=${filter_flag}
useEdgeDensity=${edgeness_flag}
useStraddling=${straddling_flag}
scalePrior=${scaleprior_flag}

trackerID=${experimentGroup}_${feature}_${kernel}_pre${pretraining}_filter${usefilter}_edge${useEdgeDensity}_straddling${useStraddling}_prior${scalePrior}

#echo $trackerID
./parallel_main --features $feature --kernel $kernel --TrackerID $trackerID --Pretraining $pretraining --useFilter $usefilter --useEdgeDensity $useEdgeDensity --useStraddling $useStraddling  --scalePrior $scalePrior
#./parallel_main
