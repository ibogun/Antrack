__author__ = 'Ivan'

import csv
import numpy as np
import string
import re
import ast
import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from  Evaluation import DatasetEvaluation


def interpolateToFixedLength(y,n=100,clean=True):

    if clean:
        goodIdx=((y<1) & (y>0))
        y=y[goodIdx]

    if len(y)==0:
        return None;

    t=np.linspace(1,y.shape[0],y.shape[0])

    f = interp1d(t,y)
    new_t=np.linspace(1,y.shape[0],n)

    y_new=f(new_t)

    return y_new


def load(file):
    csv.field_size_limit(sys.maxsize)
    reader = csv.reader(open(file, 'rb'))
    d = dict(x for x in reader)

    return d



def loadMultiDictionary(file):

    d=load(file)
    datasetDictionary=dict()
    for key,value in d.iteritems():
        #print key
        #print value

        # infinity is not good
        value = value.replace('inf', '2');

        video_dictionary= ast.literal_eval(value)
        #video_dictionary= json.loads(value)
        new_video_dictionary=dict()
        for vid_key,vid_value in video_dictionary.iteritems():
            new_video_dictionary[vid_key]=np.array(vid_value)

        datasetDictionary[key]= new_video_dictionary

    return datasetDictionary

def loadToDict(file):

    d=load(file)
    table = string.maketrans("", "")
    regex = re.compile('[%s]' % re.escape('.'))
    s = regex.sub('', string.punctuation)


    returnDict=dict()

    for key, value in d.iteritems():

        cleanString = value.translate(table, s)

        a = np.fromstring(cleanString, dtype=np.float, sep=" ")

        returnDict[key]=a;

    return returnDict








def plotObjectnessData(d,sizes, mean_xs,plotName='',savefilename='none',maxYLim=1,legend=False):
    plt.figure(figsize=(10, 7))  # Set the size of your figure, customize for more subplots
    id = 1

    n = 100
    #plt_n = len(d)/2;
    plt_n=3
    plt_m = 2;

    allColors=['g','c','m','y']

    idxCount=1
    for key, value in d.iteritems():


        xs = interpolateToFixedLength(value['origin'], n)

        ys = np.linspace(0, n - 1, n)

        plt.subplot(plt_n, plt_m, id)

        lines = list()


        l1,=plt.plot(ys, xs, linewidth=3.0, linestyle='-',label='ground truth')
        l2,=plt.plot(ys, mean_xs, marker=None, linestyle='-', color='r',label='average')

        lines.append(l1)
        lines.append(l2)

        colorId=0
        for k,v in value.iteritems():

            if k=='origin':
                continue;

            else:
                zx= interpolateToFixedLength(v, n)

                if zx is not None:
                    l,= plt.plot(ys, zx, marker=None, linestyle='-',color=allColors[colorId],label=k)

                    lines.append(l)
                colorId=colorId+1

        # X[:,id]=row
        id = id + 1;

        # for edgeness


        plt.ylim([0, maxYLim])

        plt.title(str(sizes[key])+' '+ key)
        plt.xlabel('Frame, percent')
        plt.ylabel('Prob')
        plt.grid(alpha=0.4)

        if legend and idxCount==1:
            plt.legend(handles=lines,loc=1, prop = {'size': 8})

        idxCount= idxCount+1
        # if id > plt_n * plt_m:
        #     break

    plt.tight_layout()

    if savefilename=='none':
        plt.show()
    else:
        plt.savefig(savefilename, bbox_inches='tight')

    # print names


def getSizesDictionary(dataset):
    # find average sizes of the boxes


    sizes = dict()
    for d in dataset.dictData:
        # d['boxes']?
        boxes = d['boxes']

        width_video = boxes[:, 2]
        height_video = boxes[:, 3]

        meanWidth = int(width_video.mean())
        meanHeight = int(height_video.mean())

        sizes[d["name"]] = [meanWidth, meanHeight]

    return sizes;

if __name__ == '__main__':

    wu2013results = "/Users/Ivan/Files/Results/Tracking/wu2013"
    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"

    vot2014Results = "/Users/Ivan/Files/Results/Tracking/vot2014"
    vot2014GrounTruth = "/Users/Ivan/Files/Data/vot2014"

    datasetType = 'wu2013'

    dataset = DatasetEvaluation.Dataset(wu2013GroundTruth, datasetType)

    sizes=getSizesDictionary(dataset)

    # exp_type='_multiscale';
    exp_type = '';

    straddling_filename = "straddling" + exp_type + ".csv"
    edgeness_filename = "edgeness" + exp_type + ".csv";

    plotNames = ['Straddling' + exp_type, 'Edgeness' + exp_type]

    saveFolderString='../../web/images/objectness_measures/'

    outputFileType='png' # could be pdf, png or jpeg


    measures_dictionaries=list()

    maxYLim=[1,0.5]

    measures_dictionaries.append(loadMultiDictionary(straddling_filename))
    measures_dictionaries.append(loadMultiDictionary(edgeness_filename))

    #d = loadToDict(straddling_filename)
    for d,index in zip(measures_dictionaries,range(0,len(measures_dictionaries))):

        n = 100
        m=len(d)
        # get mean first
        mean_xs = np.zeros(n)
        for value in d.itervalues():
            xs = interpolateToFixedLength(value['origin'], n)
            mean_xs = mean_xs + xs

        mean_xs = mean_xs / len(d)


        # split dictionary in parts and
        idx=1
        l=list()

        l_sizes=list()

        d1 = dict()

        d2=dict()

        names=list()
        for key,value in d.iteritems():
            names.append(key)
            d1[key] = value
            d2[key]= sizes[key]
            if idx>=6:
                idx=1
                l.append(d1)
                l_sizes.append(d2)
                d1=dict()
                d2=dict()
            else:


                idx=idx+1

        if len(d1)!=0:
            l.append(d1)

            l_sizes.append(d2)

        idx=10;

        names_idx=0

        #plotObjectnessData(l[2], l_sizes[2],mean_xs, plotNames[index] + ' ' + names[names_idx],legend=True)
        #
        #
        #break
        for i,ind_i in zip(l,range(0,len(l))):

            if ind_i==0:
                plotObjectnessData(i, l_sizes[ind_i],mean_xs, plotNames[index],
                                   savefilename=saveFolderString+plotNames[index]+str(idx)+'.'+outputFileType,
                                   maxYLim=maxYLim[index],legend=True)
            else:
                plotObjectnessData(i, l_sizes[ind_i],mean_xs, plotNames[index],
                                   savefilename=saveFolderString + plotNames[index] + str(idx) + '.' + outputFileType,
                                   maxYLim=maxYLim[index])
            idx=idx+1

            names_idx= names_idx+1



#plotObjectnessData(l[1], mean_xs, 'I  knows')

# print X
#
#
# outName="test.tsv"
#
# w=csv.writer(open(outName, 'w'), delimiter='\t')
#
# w.writerow(names)
#
# for i in range(0,n):
#     w.writerow(X[i,])