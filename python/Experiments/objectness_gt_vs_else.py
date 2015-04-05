__author__ = 'Ivan'
# Import C++ implementation of the objectness.
import sys
import math

# sys.path.append('../Experiments')
import objectness

import os
# .path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/modules')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Evaluation')
# from  Evaluation import  DatasetEvaluation
from DatasetEvaluation import Dataset, savePickle, loadPickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

import pyprind

import copy
import multiprocessing as mp


class VideoResult(object):
    """VideoResult"""

    def __init__(self, data):
        """Constructor for VideoResult"""
        self.data=data;

    def process(self):

        nSuperpixels=200
        edge_t1=1
        edge_t2=200
        inner_rectangle=0.95



        edge=list()
        straddle=list()
        print "Processing video: ", self.data["name"]
        maxNum=min(len(self.data["boxes"]), len(self.data["images"]))
        for idx in range(0, maxNum):

            image=self.data['images'][idx];
            print idx,
            objness = objectness.Objectness()
            objness.readImage(image)
            objness.initializeStraddling(nSuperpixels, inner_rectangle)
            objness.initializeEdgeDensity(edge_t1, edge_t2, inner_rectangle)

            box=self.data['boxes'][idx]

            (X,Y,W,H)=self.generateBoxes(box[0],box[1],box[2],box[3])

            straddling=objness.getStraddlingList(X,Y,W,H)
            edgeDensity=objness.getEdgenessList(X,Y,W,H)

            edge.append(edgeDensity)
            straddle.append(straddling)

        self.edge=edge
        self.straddle=straddle


    @staticmethod
    def generateBoxes(x,y,width,height,nRadial=12,nAngular=30,r=30):


        nRadialValues=np.linspace(0,r,nRadial+1)
        nAngularValues=np.linspace(0,2*np.pi,nAngular+1)

        halfWidth= width / 2.0
        halfHeight=height/2.0

        center_x=x+halfWidth
        center_y=y+halfHeight

        X=list()
        Y=list()
        W=list()
        H=list()
        X.append(int(x))
        Y.append(int(y))


        for i in range(1,len(nRadialValues)):
            for j in range(1,len(nAngularValues)):
                bb_x=center_x+(nRadialValues[i]*np.cos(nAngularValues[j]))-halfWidth
                bb_y = center_y+ (nRadialValues[i] * np.sin(nAngularValues[j])) - halfHeight

                X.append(int(bb_x))
                Y.append(int(bb_y))

        W=[int(width) for i in X]
        H=[int(height) for i in X]

        return (X,Y,W,H)


    # @staticmethod
    # def getDistances():



class DatasetResult(object):
    """"""

    def __init__(self, d):
        """Constructor for DatasetResult"""

        videos=list()

        for v in d:
            z=VideoResult(v)
            videos.append(z)

        self.videos=videos

    def process(self,cpus=3):

      for v in self.videos:
          v.process()



if __name__ == "__main__":
    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"
    datasetType = 'wu2013'

    dataset = Dataset(wu2013GroundTruth, datasetType)
    d = dataset.dictData

    wholeDataset=DatasetResult(d)
    wholeDataset.process()

    saveName='objectness_gt_vs_else_pickle.p'
    savePickle(wholeDataset,saveName)