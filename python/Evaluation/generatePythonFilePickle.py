__author__ = 'Ivan'

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

    folderNameStarsWith='b='




    subfolders = os.listdir(resultsPath);


    subfolders = [x for x in subfolders if (((not (x.startswith('.'))) and (x.startswith(folderNameStarsWith))))]
    #subfolders=['Kernelized_filter']
    for subfolder in subfolders:
        path= "/Users/Ivan/Files/Results/Tracking/wu2013/Results/"+subfolder+"/"
        type='wu2013'


        # p=['SRE','TRE']
        #
        # for pp in p:
        #     fullPath=path+pp;
        #
        #     datFiles=glob.glob(fullPath+"/*.dat")
        #
        #     for file in datFiles:
        #         print file
        #
        #         box= np.loadtxt(file, delimiter='\t')
        #
        #         box[:,0]=box[:,0]-box[:,2]/2.0
        #         box[:, 1] = box[:, 1] - box[:, 3] / 2.0
        #         box[:,[0,1,2,3]]=box[:,[1,0,3,2]]
        #         box=np.round(box)
        #         np.savetxt(file, box, delimiter=',')
        label= subfolder
        outputPath='/Users/Ivan/Code/Tracking/Antrack/python/Evaluation/Runs/'
        # #
        try:
            print path
            generatePickleFromCommandLine(path,type,label, outputPath)
        except Exception:
            print " Result for "+subfolder+" are not ready"