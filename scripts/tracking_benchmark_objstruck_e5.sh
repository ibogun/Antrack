#!/bin/bash
#SBATCH --job-name ObjAlle_e5
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=24
#SBATCH --partition=long
#SBATCH --mem-per-cpu=3000
#SBATCH --error=ObjAlle_e5_Wu2015_.%J.err
#SBATCH --output=ObjAlle_e5_Wu2015_.%J.out
#SBATCH --time=108:00:00
                                                            
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
python run_trackers_cached_parallel.py -t ObjStruck_e0_5 -p 48 
echo "Finished at `date`"

