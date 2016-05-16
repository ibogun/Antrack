#!/bin/bash
#SBATCH --job-name M64L0_4BestStruck_OPE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2000
#SBATCH --error=M64L0_4BestStruck_Wu2015_OPE_.%J.err
#SBATCH --output=M64L0_4BestStruck_Wu2015_OPE_.%J.out
#SBATCH --time=16:00:00
                                                            
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
                   
cd /udrive/student/ibogun2010/Research/Code/Antrack/                          

currentDir=$(pwd)
#cd build
#make -j8
cd $currentDir
source venv/bin/activate
cd /udrive/student/ibogun2010/Research/Code/Antrack/python/visual-tracking-benchmark
python run_trackers_cached_parallel.py -t M64L0_4BestStruck -e OPE -p 24
echo "Finished at `date`"

