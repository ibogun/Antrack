__author__ = 'Ivan'

import sys, getopt
from DatasetEvaluation import Experiment,savePickle,AllExperiments


def generatePickleFromCommandLine(results_path,datasetType,trackerLabel,picklePathPrefix='./Runs/'):

    run= AllExperiments()

    run.load(results_path,datasetType,trackerLabel)


    run.save(trackerLabel, picklePathPrefix)

if __name__ == "__main__":
    generatePickleFromCommandLine(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

    # path= "/Users/Ivan/Code/Tracking/Antrack/tmp/138f8f9a-3027-4a78-9814-6c0036295b39"
    # type='wu2013'
    #
    # label='t'
    # outputPath='/Users/Ivan/Code/Tracking/Antrack/python/Evaluation/Runs/'
    # #
    # generatePickleFromCommandLine(path,type,label, outputPath)