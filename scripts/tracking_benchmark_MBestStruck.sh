#!/bin/bash
#SBATCH --job-name MBG
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=eternity
#SBATCH --mem-per-cpu=2000
#SBATCH --error=MBG_Wu2015_.%J.err
#SBATCH --output=MBG_Wu2015_.%J.out
#SBATCH --time=408:00:00
                                                            
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
                   
cd /udrive/student/ibogun2010/Research/Code/Antrack/                          

currentDir=$(pwd)
cd build
make -j8
cd $currentDir
source venv/bin/activate
cd /udrive/student/ibogun2010/Research/Code/Antrack/python/visual-tracking-benchmark
python run_trackers_cached_parallel.py -t MBestStruckGauss -e OPE -p 24
echo "Finished at `date`"

