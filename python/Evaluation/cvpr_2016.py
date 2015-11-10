import sys
from DatasetEvaluation import Dataset, loadPickle, Evaluator
from generatePythonFilePickle import AllExperiments
from DatasetEvaluationAllExperiments import EvaluatorAllExperiments
from paperPlots import PaperPlots
import pandas as pd

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)



from  VisualizeExperiment import VisualizeExperiment
import cPickle
import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import gridspec
import glob
from matplotlib.widgets import Slider, Button
import re
import seaborn as sns
from DatasetEvaluationAllExperiments import Evaluated
sns.set(font_scale=1.9)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#sns.set_context("paper")
sns.set_style("white")
def reshuffleStraddlingEdgeOPEData(d):
    # find all
    # organize the data so that
    # s0 -> [e0 e0.1 e0.2 e0.3 e0.4 e0.5]
    # s1 -> [e0 e0.1 e0.2 e0.3 e0.4 e0.5]
    # ...

    straddling = np.tile(range(6), 6)/float(10)
    edge_density = np.repeat(range(6), 6)/float(10)

    s_list=list()

    precision=list()
    success=list()
    for i in range(0,6):
        s_row=list()
        s=0 + 0.1*i
        if i is 0:
            s='s0_'
        else:
            s = 's' +str(s)+'_'
        # get all [e0 e0.2 e0.4 etc]. Note that they are not in order
        s_row=[(key,value) for (key, value) in d.iteritems() if s in key]
        # make s_row to be in order of increasing e
        s_row_sorted=list()
        for j in range(0,6):
            e=0 +0.1*j
            if j is 0:
                e='e0.'
            else:
                e='e'+str(e)


            if j is 0:
                value = [(0 +0.1*j,x[1]) for x in s_row if x[0].endswith('0')][0]
            else:
                value = [(0 +0.1*j,x[1]) for x in s_row if e in x[0]][0]

            pr=value[1][0]
            su=value[1][1]

            if (i == 0) and (j == 0):
               su = 0.59  # These are values of the RobStruck tracker with filter on
               pr = 0.749 #
            s_row_sorted.append(value)
            precision.append(pr)
            success.append(su)


        s_list.append((0.1*i, s_row_sorted))
    df1 = pd.DataFrame(np.c_[np.asarray(precision).flat, straddling, edge_density],
                  columns=["Precision", "$\lambda_s$", "$\lambda_e$"])
    df2 = pd.DataFrame(np.c_[np.asarray(success).flat, straddling, edge_density],
                  columns=["Success", "$\lambda_s$", "$\lambda_e$"])

    return (df1, df2)


def getPrecisionSuccessGivenWildCard(paperPlot, wildcard):
    (plotMetricsDict, completeMetricDict) = paperPlot.getRuns(wildcard)
    runNames = plotMetricsDict.keys()

    regExp=re.compile("([\D|=]+)(\d+)")
    xValues=list()

    for r in runNames:
        m=regExp.match(r)
        xValues.append((float)(m.group(2)))

    idx_sorted = [i[0] for i in sorted(enumerate(xValues), key=lambda x: x[1])]

    xValues=sorted(xValues)
    correctNames = [runNames[x] for x in idx_sorted]
    p=list()
    s=list()

    for name in  correctNames:
        p.append(completeMetricDict[name][0])
        s.append(completeMetricDict[name][1])

    return (p,s, xValues)


def plotSensitivityUpd(paperPlot, baselineRun, wildcard, savefile=''):

    (p,s, xValues) = getPrecisionSuccessGivenWildCard(paperPlot, wildcard)
    run = loadPickle(baselineRun)

    baseline_p=run.completeMetricDictEntry[0]
    baseline_s = run.completeMetricDictEntry[1]

    titleFontSize = paperPlot.titleFontSize;
    headerFontSize = paperPlot.headerFontSize;
    axisFontSize = paperPlot.axisFontSize;
    lineWidth = paperPlot.lineWidth;

    legendSize = paperPlot.legendSize;
    labelsFontSize = paperPlot.labelsFontSize

    minY=paperPlot.minY
    maxY=paperPlot.maxY

    deltaPrecision=paperPlot.deltaPrecision

    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS =3
    max_yticks = 5
    # now everything is sorted
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True,  figsize=(5,4))
    fig.subplots_adjust(hspace=5)
    #plt.suptitle('Sensitivity analysis for parameter: '+wildcard,fontsize=paperPlot.axisFontSize+4)
    i  = 0
    ax[0].plot(xValues,p, paperPlot.sensitivityColorAndPoints,  linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))
    ax[0].plot([min(min(xValues),1), max(xValues)], [baseline_p, baseline_p], color='k', linestyle='--', linewidth=lineWidth)
    ax[0].set_ylim(minY+0.05+ deltaPrecision,maxY+ deltaPrecision-0.05)

    title = wildcard.replace("=",'')
    yloc = plt.MaxNLocator(max_yticks)

    ax[i].set_ylabel('Precision', color='black')
    ax[i].yaxis.set_major_locator(yloc)
        #plt.ylim((minY+ deltaPrecision,maxY+ deltaPrecision))
    #plt.ylim((minY+ deltaPrecision,maxY+ deltaPrecision))

    #ax2 = ax[0,3].twinx()
    #ax2.set_ylabel('Precision', color='black')
    #plt.xlabel(wildcard, fontsize=axisFontSize)
    #plt.ylabel("Precision", fontsize=axisFontSize)

    #plt.legend(['filter on','filter off'])
    #plt.subplot(2, 4, 5)

    ax[1].plot(xValues,s, paperPlot.sensitivityColorAndPoints,  linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))
    ax[1].plot([min(min(xValues),1), max(xValues)], [baseline_s, baseline_s], color='k', linestyle='--', linewidth=lineWidth)
    ax[1].set_ylim(minY+0.05,maxY-0.05)

    title = wildcard.replace("=",'')
    yloc = plt.MaxNLocator(max_yticks)
    i  = 0
    ax[1].set_ylabel('Success', color='black')
    ax[1].yaxis.set_major_locator(yloc)
    #for xValues, s,i in zip(xvalsList, slist, range(0,len(wildcards))):
    #    plt.plot(xValues,s, paperPlot.sensitivityColorAndPoints, linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))

    #plt.plot([min(min(xvalsList[0]), 0), max(xvalsList[0])], [baseline_s, baseline_s], color='k', linestyle='--',
    #        linewidth=lineWidth)

    #plt.ylim((minY, maxY))
    #plt.xlabel(wildcard, fontsize=axisFontSize)
    #plt.ylabel("Success", fontsize=axisFontSize)
    if savefile == '':
        plt.show()
    else:
        plt.savefig(savefile)

def plotSensitivitySpecific(paperPlot, baselineRun,wildcards,savefile=''):

    plist=list()
    slist=list()
    xvalsList=list()

    for wildcard in wildcards:
        (ps,ss, xValuess) = getPrecisionSuccessGivenWildCard(paperPlot, wildcard)
        plist.append(ps)
        slist.append(ss)
        xvalsList.append(xValuess)


    run = loadPickle(baselineRun)

    baseline_p=run.completeMetricDictEntry[0]
    baseline_s = run.completeMetricDictEntry[1]

    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(wildcards)+ 1

    titleFontSize = paperPlot.titleFontSize;
    headerFontSize = paperPlot.headerFontSize;
    axisFontSize = paperPlot.axisFontSize;
    lineWidth = paperPlot.lineWidth;

    legendSize = paperPlot.legendSize;
    labelsFontSize = paperPlot.labelsFontSize

    minY=paperPlot.minY
    maxY=paperPlot.maxY

    deltaPrecision=paperPlot.deltaPrecision

        # this should be just regular plot: value vs b

        # get values from names

    max_yticks = 5
    # now everything is sorted
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(13,5))
    #plt.suptitle('Sensitivity analysis for parameter: '+wildcard,fontsize=paperPlot.axisFontSize+4)
    for xValues, p, i in zip(xvalsList, plist, range(0,len(wildcards))):
        #ax[0,i].subplot(2,4,i+1 )
        ax[0,i].plot(xValues,p, paperPlot.sensitivityColorAndPoints,  linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))
        ax[0,i].plot([min(min(xvalsList[0]),0), max(xvalsList[0])], [baseline_p, baseline_p], color='k', linestyle='--', linewidth=lineWidth)
        ax[0,i].set_ylim(minY+0.05+ deltaPrecision,maxY+ deltaPrecision-0.05)

        title = wildcards[i].replace("=",'')
        yloc = plt.MaxNLocator(max_yticks)

        if i != 0:
            ax[0, i].get_xaxis().set_visible(False)
            ax[0,i].get_yaxis().set_visible(False)
            ax[0,i].set_title(title.upper())
        else:
            ax[0,i].set_ylabel('Precision', color='black')
            ax[0,i].set_title(title)
        ax[0,i].yaxis.set_major_locator(yloc)
        #plt.ylim((minY+ deltaPrecision,maxY+ deltaPrecision))
    #plt.ylim((minY+ deltaPrecision,maxY+ deltaPrecision))

    #ax2 = ax[0,3].twinx()
    #ax2.set_ylabel('Precision', color='black')
    #plt.xlabel(wildcard, fontsize=axisFontSize)
    #plt.ylabel("Precision", fontsize=axisFontSize)

    #plt.legend(['filter on','filter off'])
    #plt.subplot(2, 4, 5)

    for xValues, s, i in zip(xvalsList, slist, range(0,len(wildcards))):
        #ax[0,i].subplot(2,4,i+1 )
        ax[1,i].plot(xValues,s, paperPlot.sensitivityColorAndPoints,  linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))
        ax[1,i].plot([min(min(xvalsList[0]),0), max(xvalsList[0])], [baseline_s, baseline_s], color='k', linestyle='--', linewidth=lineWidth)
        ax[1,i].set_ylim(minY+0.05,maxY-0.05)
        title = wildcards[i].replace("=",'')
        yloc = plt.MaxNLocator(max_yticks)
        if i != 0:

            ax[1,i].get_yaxis().set_visible(False)
        else:
            ax[1,i].set_ylabel('Success', color='black')
        ax[1,i].yaxis.set_major_locator(yloc)
    #for xValues, s,i in zip(xvalsList, slist, range(0,len(wildcards))):
    #    plt.plot(xValues,s, paperPlot.sensitivityColorAndPoints, linewidth=lineWidth, color=cm(1.*i/NUM_COLORS))

    #plt.plot([min(min(xvalsList[0]), 0), max(xvalsList[0])], [baseline_s, baseline_s], color='k', linestyle='--',
    #        linewidth=lineWidth)

    #plt.ylim((minY, maxY))
    #plt.xlabel(wildcard, fontsize=axisFontSize)
    #plt.ylabel("Success", fontsize=axisFontSize)
    if savefile == '':
        plt.show()
    else:
        plt.savefig(savefile)

def plotSensitivity(paperPlot, baselineRun, saveResultsFolders, save):
    print "Results are averaged across all runs: TRE and SRE ( default not included)"

    wildcards = list()

    wildcards.append('b=')
    wildcards.append('r=')
    wildcards.append('q=')
    wildcards.append('p=')
    #wildcards.append('lambda')
    if save:
        for i in saveResultsFolders:

            saveResultsFolder = i + "/sensitivity.pdf"

            print "Generating figure for feature-kernel comparison...", saveResultsFolder
            plotSensitivitySpecific(paperPlot,baselineRun,wildcards,saveResultsFolder)
    else:



        plotSensitivitySpecific(paperPlot,baselineRun,wildcards)


def plotStraddelingEdgeOPE(wildcard, savefilename='', format='pdf'):
    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    folderForGraphs='./Visualizations/'
    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)
    runsNames = glob.glob("./Results/" + wildcard + "*.p")
    #runsNames = glob.glob('./Runs/upd=3_hogANDhist_int*.p')
    formatSave='pdf'
    regexp= re.compile("(.*\/)(.+)(.p)")

    experimentType='default'

    d=dict()
    for runName in runsNames:
        m=re.match(regexp,runName)
        name=m.group(2)
        run = loadPickle(runName)
        results = run.plotMetricsDictEntry['default']
        d[name]=results[4:6]

    (df1, df2)=reshuffleStraddlingEdgeOPEData(d)

    save = False
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df1, col="$\lambda_e$", hue="$\lambda_e$", col_wrap=3, size=3)

    # Draw a horizontal line to show the starting point
    grid.map(plt.axhline, y=0.749, ls=":", c=".5")

    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "$\lambda_s$", "Precision", marker="o", ms=4)
    grid.set(xticks=np.arange(6)/float(10), yticks=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                 xlim=(0, 0.5), ylim=(0.6, 0.85))
    grid.fig.tight_layout(w_pad=1)
    # Adjust the tick positions and labels

    if savefilename=='':
        plt.show()
    else:
        plt.savefig(savefilename+"precision."+format)

    grid = sns.FacetGrid(df2, col="$\lambda_e$", hue="$\lambda_e$", col_wrap=3, size=3)
    grid.map(plt.axhline, y=0.59, ls=":", c=".5")
    grid.map(plt.plot, "$\lambda_s$", "Success", marker="o", ms=4)
    grid.set(xticks=np.arange(6)/float(10), yticks=[0.5, 0.55, 0.6, 0.65, 0.7],
                 xlim=(0, 0.5), ylim=(0.5, 0.7))
    grid.fig.tight_layout(w_pad=1)

    if savefilename=='':
        plt.show()
    else:
        plt.savefig(savefilename+"success."+format)

def getRunsForCVPR2016_old():
    runs=['SAMF', 'DSST', 'upd=3_hogANDhist_int_f1']
    runsNames =['a30_hogANDhist_int_f0', 'mshogANDhist_int_f0']
    runs = runs + runsNames

    names=list()
    names.append('SAMF')
    names.append('DSST')
    names.append('RobStruck')
    names.append('ObjStruck with $\lambda_s=0, \lambda_e=0.2$')
    names.append('ObjStruck with $\lambda_s=0, \lambda_e=0.5$')
    names.append('ObjStruck with $\lambda_s=0.1, \lambda_e=0.3$')
    names.append('ObjStruck with $\lambda_s=0.2, \lambda_e=0.2$')
    names.append('ObjStruck with $\lambda_s=0.3, \lambda_e=0.4$')
    for i in range(0,len(runs)):
        runs[i]='./Results/'+runs[i]+".p"
    return (runs, names)

def getRunsForCVPR2016_robust():
    runs=['SAMF', 'DSST', 'TGPR']
    runsNames =['Rob_Struck_filter_f0','Rob_Struck_filter_f1']
    runs = runs + runsNames

    names=list()
    names.append('SAMF')
    names.append('DSST')
    names.append('TGPR')
    names.append('RobStruck filter off')
    names.append('RobStruck filter on')
    for i in range(0,len(runs)):
        runs[i]='./Results/'+runs[i]+".p"
    return (runs, names)

def getRunsForCVPR2016():
    runs=['SAMF', 'DSST', 'TGPR','Rob_Struck_filter_f1']
    runsNames =['lambda_gray_s0_e0.4', 'lambda_gray_s0.2_e0.4',
            'lambda_gray_s0.3_e0.3', 'lambda_gray_s0.4_e0.4', 'lambda_gray_s0.5_e0.3']
    runs = runs + runsNames

    names=list()
    names.append('SAMF')
    names.append('DSST')
    names.append('TGPR')
    names.append('RobStruck')
    names.append('ObjStruck with $\lambda_s=0, \lambda_e=0.4$')
    names.append('ObjStruck with $\lambda_s=0.2, \lambda_e=0.4$')
    names.append('ObjStruck with $\lambda_s=0.3, \lambda_e=0.3$')
    names.append('ObjStruck with $\lambda_s=0.4, \lambda_e=0.4$')
    names.append('ObjStruck with $\lambda_s=0.5, \lambda_e=0.3$')
    for i in range(0,len(runs)):
        runs[i]='./Results/'+runs[i]+".p"
    return (runs, names)


def plot_OPE_comparison():
    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)

    (runsNames, corrected_names)= getRunsForCVPR2016()

    runs = list()
    names=list()
    for runName in runsNames:
        run = loadPickle(runName)
        names.append(runName)
        runs.append(run)

    plotMetricsDict=dict()
    completeMetricDict=dict()

    alternativeNames=corrected_names
    if len(alternativeNames) == 0:
        for r in runs:
            plotMetricsDict[r.name]=r.plotMetricsDictEntry
            completeMetricDict[r.name]=r.completeMetricDictEntry
    else:
        for r,name in zip(runs,alternativeNames):
            plotMetricsDict[name]=r.plotMetricsDictEntry
            completeMetricDict[name]=r.completeMetricDictEntry

    success=list()
    precision=list()
    names=list()
    import re
    for name, value in plotMetricsDict.iteritems():
        s=value['default'][5]
        p=value['default'][4]
        success.append(s)
        precision.append(p)
        if "ObjStruck with " in name:
            short_name=re.sub("ObjStruck with ",'',name)
        else:
            short_name=name
        names.append(short_name)

    df = pd.DataFrame({'name': names, 'success': success, 'precision':precision})
    print df

    sns.set_style("whitegrid")
    plt.subplots_adjust(bottom=0.4)
    ax=sns.barplot(x="name", y="success", data=df,palette="Blues_d")
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=-45)

    plt.show()

def plotOPE_SRE_TRE_Robust(saveFigureToFolder = '/Users/Ivan/Code/Tracking/Antrack/doc/technical_reports/images/',
                    format='png'):
    wu2013results = "/Users/Ivan/Files/Results/Tracking/wu2013"
    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"

    vot2014Results = "/Users/Ivan/Files/Results/Tracking/vot2014"
    vot2014GrounTruth = "/Users/Ivan/Files/Data/vot2014"

    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)

    (runsNames, corrected_names)= getRunsForCVPR2016_robust()

    runs = list()
    names=list()
    for runName in runsNames:
        run = loadPickle(runName)
        names.append(runName)
        #run = run.data[experimentType]
        runs.append(run)

    # run=loadPickle('./Runs/TLD.p')
    # runs.append(run)
    # names.append('./Runs/TLD.p')

    save = True
    evaluator = EvaluatorAllExperiments(dataset, list(), names)

    successAndPrecision = 'cvpr2016_SuccessAndPrecision_wu2013'
    histograms = 'cvpr2016_histogram_wu2013'


    i = format
    if save:
        evaluator.evaluateFromSave(runs,successAndPrecisionPlotName=saveFigureToFolder+successAndPrecision+'.'+
                                                        i,histogramPlot=saveFigureToFolder+histograms+'.'+
                                                                                       i, alternativeNames=corrected_names)
    else:
        evaluator.evaluateFromSave(runs, alternativeNames=corrected_names)

def plotOPE_SRE_TRE(saveFigureToFolder = '/Users/Ivan/Code/Tracking/Antrack/doc/technical_reports/images/',
                    format='png'):
    wu2013results = "/Users/Ivan/Files/Results/Tracking/wu2013"
    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"

    vot2014Results = "/Users/Ivan/Files/Results/Tracking/vot2014"
    vot2014GrounTruth = "/Users/Ivan/Files/Data/vot2014"

    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)

    (runsNames, corrected_names)= getRunsForCVPR2016()

    runs = list()
    names=list()
    for runName in runsNames:
        run = loadPickle(runName)
        names.append(runName)
        #run = run.data[experimentType]
        runs.append(run)

    # run=loadPickle('./Runs/TLD.p')
    # runs.append(run)
    # names.append('./Runs/TLD.p')


    evaluator = EvaluatorAllExperiments(dataset, list(), names)

    successAndPrecision = 'cvpr2016_SuccessAndPrecision_wu2013'
    histograms = 'cvpr2016_histogram_wu2013'


    i = format
    evaluator.evaluateFromSave(runs,successAndPrecisionPlotName=saveFigureToFolder+successAndPrecision+'.'+
                                                    i,histogramPlot=saveFigureToFolder+histograms+'.'+
                                                                                   i, alternativeNames=corrected_names)
        #evaluator.evaluateFromSave(runs,alternativeNames=corrected_names)

    #evaluator.evaluateFromSave(runs, alternativeNames=corrected_names)

def robustKalman():
    savefilename="/Users/Ivan/Documents/Papers/My_papers/CVPR_2016_Robust_tracking/images/"
    format='png'

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
    paperPlot=PaperPlots(dataset,folder)

    # this has to changeop
    baseLineRun= folder+'Rob_Struck_filter_f0'+".p"

    plotOPE_SRE_TRE_Robust(saveFigureToFolder=savefilename, format='pdf')
    plotSensitivity(paperPlot,baseLineRun,saveResultsFolder,save)
    #saveUpdName =saveResultsFolder+"/update.pdf"
    #plotSensitivityUpd(paperPlot,baseLineRun,"upd=",savefile='')
def objectAware():
    savefilename="/Users/Ivan/Code/Tracking/Antrack/doc/technical_reports/images/"
    savefilename="/Users/Ivan/Documents/Papers/My_papers/CVPR_2016_Object-aware_tracking/images/"
    format='png'
    plotStraddelingEdgeOPE("lambda_gray", savefilename=savefilename+'straddeling_edge_OPE',format=format)
    plotOPE_SRE_TRE(saveFigureToFolder=savefilename)
    #plot_OPE_comparison()

if __name__ == "__main__":
    robustKalman()
    objectAware()