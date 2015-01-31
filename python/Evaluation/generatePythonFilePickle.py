__author__ = 'Ivan'

import sys, getopt
from DatasetEvaluation import Experiment,savePickle
def generatePickleFromCommandLine(results_path,datasetType,trackerLabel):

    run=Experiment(results_path,datasetType,trackerLabel)
    run.loadResults()


    picklePath='./Runs/'+trackerLabel+'.p'

    savePickle(run,picklePath)

if __name__ == "__main__":
    generatePickleFromCommandLine(sys.argv[1],sys.argv[2],sys.argv[3])