__author__ = 'Ivan'

import numpy as np
from scipy import io
import glob
import re
from DatasetEvaluation import Experiment,AllExperiments,savePickle

class MatLoader(object):

    def __init__(self,datasetType,trackerName):
        self.datasetType=datasetType
        self.trackerName= trackerName

        self.allExperiments= AllExperiments()

        self.allExperiments.data=dict()


    def findAllVideos(self,path):
        runsNames = glob.glob(path+"/"+"*"+self.trackerName+".mat");

        regExpression = re.compile("(.*\/+)([\w|-]+)(?=_.*)")
        # list of (videoName,list of runs)

        regExpressionForTRE=re.compile(".*TRE.*")

        data=list()
        opeExperiments = list()

        match1 = re.match(regExpressionForTRE, runsNames[0])

        isTRE = (match1 is not None)

        for name in runsNames:

            match=re.match(regExpression,name)
            videoName=match.group(2)

            videoName=videoName.lower()

            print videoName

            mat=io.loadmat(name)
            mat=mat['results'][0];


            nExperiments=list()

            if isTRE:
                rns=list()

                array= mat[0][0][0][1]
                #array.astype(np.int64)
                rns.append(array)
                opeExperiments.append((videoName, rns))



            for m in mat:


                run=m[0]
                run=run[0]
                boxes=run[1]

                boxes.astype(np.int64)
                if len(boxes[0])==6:
                    raise NotImplementedError(" bounding boxes which are not given by (left corner x, left corner y, width, height)"
                                              "are not implemented")

                nExperiments.append(boxes)
            data.append((videoName,nExperiments));

        run = Experiment(path, self.datasetType, self.trackerName)
        run.data=data;


        if isTRE:
            self.allExperiments.data['TRE']=run

            r=Experiment(path,self.datasetType,self.trackerName)
            r.data=opeExperiments;

            self.allExperiments.data['default']=r
        else:
            self.allExperiments.data['SRE']=run;


    def getAllExperiments(self):

        return self.allExperiments;


    def checkConsistency(self):


        for expName,exp in self.allExperiments.data.iteritems():

            if len(exp.data)!=51:
                print "Current experiment name: ",expName
                raise Exception("got wrong size: expected 50, got "+str(len(exp.data)))



if __name__ == "__main__":

    folderWithWu2013Results='/Users/Ivan/Files/Data/wu2013/results/'
    saveLocation='./Runs/'

    treResults=folderWithWu2013Results+"results_TRE_CVPR13/"
    sreResults=folderWithWu2013Results+"results_SRE_CVPR13/"

    datasetType='wu2013'


    trackers=list()
    trackers.append('TLD')
    trackers.append('Struck')


    for trackerName in trackers:

        print trackerName
        loader=MatLoader(datasetType,trackerName)

        loader.findAllVideos(treResults)
        loader.findAllVideos(sreResults)

        exp=loader.getAllExperiments()
        loader.checkConsistency()
        exp.save(trackerName)
    # save exp