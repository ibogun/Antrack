__author__ = 'Ivan'
import traceback
import sys, getopt,os
from DatasetEvaluation import Experiment,savePickle,AllExperiments

import glob
import numpy as np
def generatePickleFromCommandLine(results_path,datasetType,trackerLabel,picklePathPrefix='./Runs/'):

    run= AllExperiments()

    run.load(results_path,datasetType,trackerLabel)


    run.save(trackerLabel, picklePathPrefix)

if __name__ == "__main__":
    #generatePickleFromCommandLine(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

    resultsPath="/Users/Ivan/Files/Results/Tracking/wu2013/Results/"

    folderNameStarsWith='Rob_'




    subfolders = os.listdir(resultsPath);


    subfolders = [x for x in subfolders if (((not (x.startswith('.'))) and (x.startswith(folderNameStarsWith))))]
    #subfolders=['SAMF']
    for subfolder in subfolders:
        path= "/Users/Ivan/Files/Results/Tracking/wu2013/Results/"+subfolder+"/"
        type='wu2013'

        label= subfolder
        outputPath='/Users/Ivan/Code/Tracking/Antrack/python/Evaluation/Runs/'
        # #
        try:
            print path
            generatePickleFromCommandLine(path,type,label, outputPath)
        except Exception:
            traceback.print_exc()
            print " Result for "+subfolder+" are not ready"
