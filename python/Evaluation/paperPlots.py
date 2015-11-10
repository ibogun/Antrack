__author__ = 'Ivan'

from DatasetEvaluation import Dataset, loadPickle, AllExperiments, Evaluator, savePickle
from DatasetEvaluationAllExperiments import Evaluated,EvaluatorAllExperiments
import glob

import matplotlib.pyplot as plt
import numpy as np

import re
class PaperPlots(object):
    """"""
    
    def __init__(self, dataset,folder):
        """Constructor for PaperPlots"""

        self.dataset = dataset
        self.folder = folder

        self.cm = plt.get_cmap('gist_rainbow')

        self.titleFontSize = 16;
        self.headerFontSize = 14;
        self.axisFontSize = 16;
        self.lineWidth = 2.8;

        self.legendSize = 6;
        self.labelsFontSize = 7

        self.minY = 0.47
        self.maxY = 0.65

        self.deltaPrecision = 0.18

        self.sensitivityColorAndPoints='bo-'


    def getRuns(self,wildcard):
        runsNames = glob.glob(self.folder + wildcard + '*.p')

        runs = list()
        #
        names = list()
        for runName in runsNames:
            run = loadPickle(runName)
            names.append(runName)
            # run = run.data[experimentType]
            runs.append(run)


        plotMetricsDict = dict()
        completeMetricDict = dict()

        for r in runs:
            plotMetricsDict[r.name] = r.plotMetricsDictEntry
            completeMetricDict[r.name] = r.completeMetricDictEntry

        return (plotMetricsDict,completeMetricDict)


    def plotsensitivity(self, baselineRun,wildcard,savefile=''):

        (plotMetricsDict, completeMetricDict) = self.getRuns(wildcard)

        run = loadPickle(baselineRun)

        baseline_p=run.completeMetricDictEntry[0]
        baseline_s = run.completeMetricDictEntry[1]

        cm = plt.get_cmap('gist_rainbow')

        titleFontSize = self.titleFontSize;
        headerFontSize = self.headerFontSize;
        axisFontSize = self.axisFontSize;
        lineWidth = self.lineWidth;

        legendSize = self.legendSize;
        labelsFontSize = self.labelsFontSize


        minY=self.minY
        maxY=self.maxY

        deltaPrecision=self.deltaPrecision

        # this should be just regular plot: value vs b

        # get values from names
        runNames = plotMetricsDict.keys()

        regExp=re.compile("([\D|=]+)(\d+)")


        xValues=list()

        for r in runNames:

            m=regExp.match(r)

            xValues.append((float)(m.group(2)))

        # sort xValues

        idx_sorted = [i[0] for i in sorted(enumerate(xValues), key=lambda x: x[1])]

        xValues=sorted(xValues)
        correctNames = [runNames[x] for x in idx_sorted]

        # now everything is sorted


        plt.figure(figsize=(13,9))



        p=list()
        s=list()

        for name in  correctNames:
            p.append(completeMetricDict[name][0])
            s.append(completeMetricDict[name][1])


        plt.suptitle('Sensitivity analysis for parameter: '+wildcard,fontsize=self.axisFontSize+4)

        plt.subplot(1,2,1)

        plt.plot(xValues,p, self.sensitivityColorAndPoints,linewidth=lineWidth)
        plt.plot([min(min(xValues),0), max(xValues)], [baseline_p, baseline_p], color='k', linestyle='--', linewidth=lineWidth)

        plt.ylim((minY+ deltaPrecision,maxY+ deltaPrecision))
        #plt.title("Precision")

        plt.xlabel(wildcard, fontsize=axisFontSize)
        plt.ylabel("Precision", fontsize=axisFontSize)

        plt.legend(['filter on','filter off'])
        plt.subplot(1, 2, 2)

        plt.plot(xValues,s, self.sensitivityColorAndPoints, linewidth=lineWidth)

        plt.plot([min(min(xValues), 0), max(xValues)], [baseline_s, baseline_s], color='k', linestyle='--',
                 linewidth=lineWidth)

        plt.ylim((minY, maxY))
        #plt.title("Success")

        plt.xlabel(wildcard, fontsize=axisFontSize)
        plt.ylabel("Success", fontsize=axisFontSize)
        if savefile == '':
            plt.show()
        else:
            plt.savefig(savefile)



    def plotFeatureKernel(self,savefile=''):

        wildcard='fk'
        (plotMetricsDict, completeMetricDict)=self.getRuns(wildcard)

        # group metrics so that feature-kernel f0-f1
        # actually they are grouped already


        precision = list()
        success = list()
        cm = plt.get_cmap('gist_rainbow')

        titleFontSize = 16;
        headerFontSize = 12;
        axisFontSize = 12;
        lineWidth = 1.8;

        legendSize = 6;

        labelsFontSize = 7
        idx = 1

        #import seaborn as sns

        p = list()
        s = list()

        runNames=plotMetricsDict.keys()


        runNamesWithFilter=[x for x in runNames if x.endswith("_f1")]
        runN= [x for x in runNames if x.endswith("_f0")]

        # now lets make them in order


        runNamesWithoutFilter=list()


        p_f0=list()
        p_f1=list()

        s_f0=list()
        s_f1=list()

        for name in runNamesWithFilter:


            correspondingName=[x for x in runN if x[0:len(x)-3]==name[0:len(x)-3]][0]
            runNamesWithoutFilter.append(correspondingName)

            p_f0.append(completeMetricDict[correspondingName][0])
            p_f1.append(completeMetricDict[name][0])

            s_f0.append(completeMetricDict[correspondingName][1])
            s_f1.append(completeMetricDict[name][1])


        groups = np.arange(len(p))

        niceNames=[x[3:len(x)-3] for x in runNamesWithFilter]

        plt.figure(figsize=(13,9))
        #fig, ax = plt.subplots()

        n_groups = len(p_f1)
        index = np.arange(n_groups)
        bar_width = 0.35

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        plt.subplot(1,2,1)
        rects1 = plt.bar(index, p_f0, bar_width,
                         alpha=opacity,
                         color='b',
                         error_kw=error_config,
                         label='f0')

        #plt.xlim((0, 1))

        rects2 = plt.bar(index + bar_width, p_f1, bar_width,
                         alpha=opacity,
                         color='r',
                         error_kw=error_config,
                         label='f1')
        plt.ylim((0, 1))
        #plt.xlabel('Group')
        #plt.ylabel('Scores')
        plt.title('Precision',fontsize=headerFontSize)
        plt.xticks(index + bar_width, niceNames)
        plt.legend()

        plt.subplot(1, 2, 2)
        rects1 = plt.bar(index, s_f0, bar_width,
                         alpha=opacity,
                         color='b',
                         error_kw=error_config,
                         label='f0')

        # plt.xlim((0, 1))

        rects2 = plt.bar(index + bar_width, s_f1, bar_width,
                         alpha=opacity,
                         color='r',
                         error_kw=error_config,
                         label='f1')
        plt.ylim((0, 1))
        # plt.xlabel('Group')
        #plt.ylabel('Scores')
        plt.title('Success', fontsize=headerFontSize)
        plt.xticks(index + bar_width, niceNames)
        plt.legend()

        plt.tight_layout()

        if savefile=='':
            plt.show()
        else:
            plt.savefig(savefile)



def plotComparisonToOtherTrackers(dataset, saveFigureToFolders,save):


    # TODO: Add TLD tracker and, perhaps, SCM
    runsNames = ['SAMF','DSST', 'upd=3_hogANDhist_int_f1','a30_hogANDhist_int_f1']
    runs = list()
    #
    names = list()
    for runName in runsNames:
        runName = './Results/' + runName + '.p'
        run = loadPickle(runName)
        names.append(runName)
        print runName
        # run = run.data[experimentType]
        runs.append(run)

    # run=loadPickle('./Runs/TLD.p')
    # runs.append(run)
    # names.append('./Runs/TLD.p')


    evaluator = EvaluatorAllExperiments(dataset, list(), names)

    # saveFormat = ['png', 'pdf']
    saveFormat = ['eps']
    successAndPrecision = 'SuccessAndPrecision_wu2013'
    histograms = 'histogram_wu2013'


    if save:
        for folder in saveFigureToFolders:
            evaluator.evaluateFromSave(runs,successAndPrecisionPlotName=folder+"/"+successAndPrecision+'.'+
                                                           saveFormat[0],histogramPlot=folder+ "/"+histograms+'.'+
                                                                                       saveFormat[0])
    else:
        evaluator.evaluateFromSave(runs)

def plotFeatureKernel(paperPlot, saveResultsFolders,save):


    print "Results are averaged across all runs: TRE and SRE ( default not included)"

    if save:
        for i in saveResultsFolders:


            saveResultsFolder = i + "/feature_kernel.pdf"

            print "Generating figure for feature-kernel comparison...", saveResultsFolder
            paperPlot.plotFeatureKernel(saveResultsFolder)
    else:
        paperPlot.plotFeatureKernel()


def plotSensitivity(paperPlot,baselineRun, saveResultsFolders, save):
    print "Results are averaged across all runs: TRE and SRE ( default not included)"

    wildcards = list()

    #wildcards.append('r')
    #wildcards.append('b')
    #wildcards.append('q')
    #wildcards.append('p')
    wildcards.append('lambda')
    if save:
        for i in saveResultsFolders:

            for w in wildcards:
                saveResultsFolder = i + "/"+w+"_sensitivity.pdf"

                print "Generating figure for feature-kernel comparison...", saveResultsFolder
                paperPlot.plotsensitivity(baselineRun,w,saveResultsFolder)
    else:



        for w in wildcards:
            paperPlot.plotsensitivity(baselineRun,w)


def plotParameter(paperPlot,baselineRun, saveResultsFolders, save, parameter='upd'):
    print "Results are averaged across all runs: TRE and SRE ( default not included)"

    wildcards = list()
    wildcards.append(parameter)
    if save:
        for i in saveResultsFolders:

            for w in wildcards:
                saveResultsFolder = i + "/"+w+"_sensitivity.pdf"

                print "Generating figure for parameter ",parameter," comparison...", saveResultsFolder
                paperPlot.plotsensitivity(baselineRun,w,saveResultsFolder)
    else:



        for w in wildcards:
            paperPlot.plotsensitivity(baselineRun,w)



def customParameter():

    folder='./Results/'

    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    saveResultsFolder=list()

    bookChapter="/Users/Ivan/Documents/Papers/My_papers/Tracking_book_chapter/images"
    paper="/Users/Ivan/Documents/Papers/My_papers/Tracking_with_Robust_Kalman/images"

    saveResultsFolder.append(bookChapter)
    saveResultsFolder.append(paper)



    save=False

    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)


    baseLineRun= folder+'upd=3_hogANDhist_int_f1'+".p"


    paperPlot=PaperPlots(dataset,folder)

    #plotComparisonToOtherTrackers(dataset,saveResultsFolder,save)
    #plotFeatureKernel(paperPlot,saveResultsFolder,save)
    plotParameter(paperPlot, baseLineRun,saveResultsFolder,save, parameter='lambda')

def main():

    folder='./Results/'

    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    saveResultsFolder=list()

    bookChapter="/Users/Ivan/Documents/Papers/My_papers/Tracking_book_chapter/images"
    paper="/Users/Ivan/Documents/Papers/My_papers/CVPR_2016_Robust_tracking/images"

    #saveResultsFolder.append(bookChapter)
    saveResultsFolder.append(paper)
    save=True

    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)


    baseLineRun= folder+'upd=1_hogANDhist_int_f1'+".p"


    paperPlot=PaperPlots(dataset,folder)

    plotComparisonToOtherTrackers(dataset,saveResultsFolder,save)
    #plotFeatureKernel(paperPlot,saveResultsFolder,save)
    plotSensitivity(paperPlot, baseLineRun,saveResultsFolder,save)


if __name__ == "__main__":
    main()
    #customParameter()
