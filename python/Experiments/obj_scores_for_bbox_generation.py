__author__ = 'Ivan'
# Import C++ implementation of the objectness.
import sys
import math

#sys.path.append('../Experiments')
import objectness

import os
# .path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/modules')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Evaluation')
# from  Evaluation import  DatasetEvaluation
from DatasetEvaluation import Dataset,savePickle,loadPickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

import pyprind

import copy
import multiprocessing as mp

def processOneImage(imageName, box,minScale,maxScale,downsample, R=60):
    objness = objectness.Objectness()

    nSuperpixels = 200
    inner_rectangle = 0.95
    edge_t1 = 1
    edge_t2 = 200
    objness.readImage(imageName)
    objness.initializeStraddling(nSuperpixels, inner_rectangle)
    objness.initializeEdgeDensity(edge_t1, edge_t2, inner_rectangle)

    x = int(box[0] + box[2] / 2.0)
    y = int(box[1] + box[3] / 2.0)

    x_axis, y_axis = np.ogrid[x - R:x + R, y - R:y + R]
    mask = (x_axis - x) * (x_axis - x) + (y_axis - y) * (y_axis - y) <= R * R

    thirdDim = maxScale - minScale + 1
    O_edge = np.zeros((len(x_axis), len(y_axis), thirdDim))
    O_straddling = np.zeros((len(x_axis), len(y_axis), thirdDim))
    O_both = np.zeros((len(x_axis), len(y_axis), thirdDim))

    listX = list()
    listY = list()
    listW = list()
    listH = list()

    listIdx = list()

    for i in range(x - R, x + R):
        for j in range(y - R, y + R):

            if mask[i - x][j - y]:
                for s in range(minScale, maxScale + 1):
                    width = int(box[2] * np.power(downsample, s));
                    height = int(box[3] * np.power(downsample, s));
                    # make the box and perform calculations

                    b = [i, j, width, height]

                    listX.append(i)
                    listY.append(j)
                    listW.append(width)
                    listH.append(height)
                    listIdx.append([i, j, s])

    # perform objectness by passing the whole list
    obj_edgeness = objness.getEdgenessList(listX, listY, listW, listH)
    obj_straddling = objness.getStraddlingList(listX, listY, listW, listH)

    obj_combined = [x * y for x, y in zip(obj_straddling, obj_edgeness)]

    outlist=list()

    currentScale=lambda obj_list,tupleList,scale: [(x, y) for (x, y) in zip(obj_list, tupleList) if y[2] == scale];

    out_edge_list=list()
    out_straddling_list=list()
    out_cobined_list=list()

    for s in range(minScale, maxScale + 1):

        currentIdx= currentScale(listIdx, listIdx, s)


        # get only elements from the scale s
        current=currentScale(obj_edgeness,listIdx,s)
        bestIdx= current.index(max(current))
        out_edge_list.append(currentIdx[bestIdx])

        current = currentScale(obj_straddling, listIdx, s)
        bestIdx = current.index(max(current))
        out_straddling_list.append(currentIdx[bestIdx])

        current = currentScale(obj_combined, listIdx, s)
        bestIdx = current.index(max(current))
        out_cobined_list.append(currentIdx[bestIdx])

    # for every s find maximum over the other two and return those

    return (imageName, out_edge_list, out_straddling_list, out_cobined_list)

class ObjectnessBoxesEvaluator(object):
    def __init__(self, dataset,cpus=4, downsample=1.05, minScale=-4, maxScale=10):
        self.dataset = dataset
        self.minScale = minScale
        self.maxScale = maxScale
        self.downsample = downsample

        self.cpus=cpus;

    def evaluate(self):
        """Evaluate objectness scores

        Args:


        Returns:

        """

        output = list()
        n=len(self.dataset.dictData)
        bar=pyprind.ProgBar(n,monitor=True,title='Objectness measures as trackers')

        idx=0
        for video in self.dataset.dictData:
            # video is dictionary : 'images'-> ims, 'boxes'->np.array,'name'->vidName

            # evaluate video here

            # in the end
            output=self.evaluateVideo(video)

            output.append((video['name'], output))

            bar.update()
            idx=idx+1
            #
            #
            # if idx>=3:
            #     break;



        self.output=output;

    def evaluateVideo(self, video, R=60):
        # video is dictionary : 'images'-> ims, 'boxes'->np.array,'name'->vidName

        #maxImages = 8
        #
        #video['images'] = video['images'][:maxImages]
        #video['boxes'] = video['boxes'][:maxImages]

        pool=mp.Pool(processes=self.cpus)
        #results = pool.map(processOneImage, zip(video['images'], video['boxes']))
        results = [pool.apply_async(processOneImage, args=(image,box, self.minScale, self.maxScale, self.downsample, R )) for image,box in
                   zip(video['images'], video['boxes'])]
        output = [p.get() for p in results]


        return output


    def save(self,savefile):

        savePickle(self,savefile)


def main():
    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"
    datasetType = 'wu2013'

    dataset = Dataset(wu2013GroundTruth, datasetType)
    d = dataset.dictData

    maxCpus = 4
    obj = ObjectnessBoxesEvaluator(dataset, cpus=maxCpus)

    obj.evaluate()

if __name__ == "__main__":
    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"
    datasetType = 'wu2013'

    dataset = Dataset(wu2013GroundTruth, datasetType)
    d = dataset.dictData

    maxCpus = 4

    saveTo="objectness_as_tracker"

    obj = ObjectnessBoxesEvaluator(dataset, cpus=maxCpus)

    obj.evaluate()

    obj.save(savefile=saveTo)
