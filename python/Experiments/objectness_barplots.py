__author__ = 'Ivan'

from clean_data import loadMultiDictionary,interpolateToFixedLength
from  Evaluation import DatasetEvaluation
from clean_data import getSizesDictionary
import numpy as np

from scipy.integrate import simps
from matplotlib import pyplot as plt


def plotAverages(dictionary,title='No Title',savefilename=''):

    #dictionary=dict()

    experiment_names=list()
    for value in dictionary.itervalues():
        num_experiments=len(value)

        for key in value.iterkeys():
            experiment_names.append(key)

    n = 100
    X=np.zeros((num_experiments,n))

    counts=np.zeros((num_experiments))

    for video_dictionary in dictionary.itervalues():

        for key,index in zip(video_dictionary.iterkeys(),range(0,num_experiments)):
            xs = interpolateToFixedLength(video_dictionary[key], n)

            if xs is not None:
                X[index,]= X[index,]+xs
                counts[index]= counts[index]+1

    y = np.linspace(0, 1, n)

    values = np.zeros((num_experiments))

    for i in range(0,num_experiments):
        z=X[i,]/counts[i]

        values[i]=simps(z,y)

    fig = plt.figure()
    sort_index = np.argsort(values)
    values=values[sort_index]



    labels=list()

    for i in range(0,len(sort_index)):
        labels.append(experiment_names[sort_index[i]])


    values= values[::-1]

    labels=list(reversed(labels))

    width = 0.3
    ind = np.arange(len(values))
    plt.bar(ind, values,align="center")
    plt.xticks(ind, labels)
    plt.title(title)
    #fig.autofmt_xdate()

    if savefilename!='':
        plt.savefig(savefilename)
    else:
        plt.show()



def plotAreaVsStraddling(straddling_dictionary, sizes, savefilename=''):

    X=np.zeros((len(sizes)))
    Y=np.zeros((len(sizes)))
    idx=0
    for name, video in straddling_dictionary.iteritems():

        dims=sizes[name]

        X[idx]=dims[0]*dims[1]

        measure= interpolateToFixedLength(video['origin'])
        Y[idx]=measure.mean()

        idx=idx+1

    plt.scatter(X,Y)
    plt.title('Effect of the size ')
    plt.xlabel('Area, pixels')
    plt.ylabel('Average straddling measure')
    plt.xlim([0, 30000])

    if savefilename=='':
        plt.show()
    else:
        plt.savefig(savefilename)

if __name__ == '__main__':
    straddling_filename = "straddling.csv"
    edgeness_filename = "edgeness.csv";

    plotNames = ['Straddling', 'Edgeness']

    saveFolderString = '../../web/images/objectness_measures/'

    outputFileType = 'png'  # could be pdf, png or jpeg

    measures_dictionaries = list()

    maxYLim = [1, 0.5]

    measures_dictionaries.append(loadMultiDictionary(straddling_filename))
    measures_dictionaries.append(loadMultiDictionary(edgeness_filename))

    for i in range(0,len(measures_dictionaries)):
        plotAverages(measures_dictionaries[i],plotNames[i],saveFolderString+'Bar_'+plotNames[i]+'.'+outputFileType)


    wu2013results = "/Users/Ivan/Files/Results/Tracking/wu2013"
    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"

    vot2014Results = "/Users/Ivan/Files/Results/Tracking/vot2014"
    vot2014GrounTruth = "/Users/Ivan/Files/Data/vot2014"

    datasetType = 'wu2013'

    dataset = DatasetEvaluation.Dataset(wu2013GroundTruth, datasetType)

    sizes = getSizesDictionary(dataset)

    straddling_dictionary=measures_dictionaries[0]
    plotAreaVsStraddling(straddling_dictionary,sizes,savefilename=saveFolderString+'area_vs_straddling.png')