#!/bin/bash
#SBATCH --job-name DeepStruck_OPE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --error=DeepStruck_OPE_Wu2015_.%J.err
#SBATCH --output=DeepStruck_OPE_Wu2015_.%J.out
#SBATCH --time=216:00:00
                                                            
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
                   
cd /udrive/student/ibogun2010/Research/Code/Antrack/                          

module unload gcc
currentDir=$(pwd)
#cd build
#make -j8
cd $currentDir
module load cuda
export PYTHONPATH=/udrive/student/ibogun2010/Download_glacier/caffe-rc3/python::/usr/lib64/python2.6/site-packages:/udrive/student/ibogun2010/local/lib/python2.7/site-packages:/udrive/student/ibogun2010/local/include/python2.7

export LD_LIBRARY_PATH=/udrive/student/ibogun2010/local_glacier/lib:$LD_LIBRARY_PATH

export PATH=/udrive/student/ibogun2010/local_glacier/bin:$PATH

caffe device_query -gpu 0
cd /udrive/student/ibogun2010/Research/Code/Antrack/python/visual-tracking-benchmark
python run_trackers_cached.py -t DeepStruck -e OPE
echo "Finished at `date`"

