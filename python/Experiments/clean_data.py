__author__ = 'Ivan'

import csv
import numpy as np
import string
import re

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def interpolateToFixedLength(y,n=100,clean=True):

    if clean:
        goodIdx=((y<1) & (y>0))
        y=y[goodIdx]


    t=np.linspace(1,y.shape[0],y.shape[0])

    f = interp1d(t,y)
    new_t=np.linspace(1,y.shape[0],n)

    y_new=f(new_t)

    return y_new


def load(file):
    reader = csv.reader(open(file, 'rb'))
    d = dict(x for x in reader)

    return d





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








def plotObjectnessData(d, mean_xs,plotName='',savefilename='none',maxYLim=1):
    plt.figure(figsize=(10, 7))  # Set the size of your figure, customize for more subplots
    id = 1

    n = 100
    #plt_n = len(d)/2;
    plt_n=3
    plt_m = 2;


    for key, value in d.iteritems():


        xs = interpolateToFixedLength(value, n)

        ys = np.linspace(0, n - 1, n)

        plt.subplot(plt_n, plt_m, id)
        l1,=plt.plot(ys, xs, marker='o',label='z')
        l2,=plt.plot(ys, mean_xs, marker=None, linestyle='-', color='r',label='average')
        # X[:,id]=row
        id = id + 1;

        # for edgeness


        plt.ylim([0, maxYLim])

        plt.title(plotName)
        plt.xlabel('Frame, percent')
        plt.ylabel('Prob')
        plt.grid(alpha=0.4)
        #plt.legend(handles=[l2],loc=1)
        # if id > plt_n * plt_m:
        #     break

    plt.tight_layout()

    if savefilename=='none':
        plt.show()
    else:
        plt.savefig(savefilename, bbox_inches='tight')

    # print names

if __name__ == '__main__':


    straddling_filename = "straddling.csv"
    edgeness_filename = "edgeness.csv";

    plotNames=['Straddling', 'Edgeness']

    saveFolderString='../../web/images/objectness_measures/'

    outputFileType='png' # could be pdf, png or jpeg


    measures_dictionaries=list()

    maxYLim=[1,0.5]

    measures_dictionaries.append(loadToDict(straddling_filename))
    measures_dictionaries.append(loadToDict(edgeness_filename))

    #d = loadToDict(straddling_filename)
    for d,index in zip(measures_dictionaries,range(0,len(measures_dictionaries))):

        n = 100
        m=len(d)
        # get mean first
        mean_xs = np.zeros(n)
        for value in d.itervalues():
            xs = interpolateToFixedLength(value, n)
            mean_xs = mean_xs + xs

        mean_xs = mean_xs / len(d)


        # split dictionary in parts and
        idx=1
        l=list()

        d1 = dict()

        names=list()
        for key,value in d.iteritems():
            names.append(key)
            d1[key] = value
            if idx>=6:
                idx=1
                l.append(d1)
                d1=dict()
            else:


                idx=idx+1

        if len(d1)!=0:
            l.append(d1)

        idx=10;

        names_idx=0

        # plotObjectnessData(l[0],mean_xs)
        # #
        # #
        # break
        for i in l:
            plotObjectnessData(i,mean_xs, plotNames[index]+' '+ names[names_idx],
                               savefilename=saveFolderString+plotNames[index]+str(idx)+'.'+outputFileType,
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