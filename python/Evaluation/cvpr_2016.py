import sys
from DatasetEvaluation import Dataset, loadPickle, Evaluator
from generatePythonFilePickle import AllExperiments
from DatasetEvaluationAllExperiments import EvaluatorAllExperiments

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
sns.set(font_scale=1.7)
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

            s_row_sorted.append(value)
            precision.append(value[1][0])
            success.append(value[1][1])


        s_list.append((0.1*i, s_row_sorted))
    df1 = pd.DataFrame(np.c_[np.asarray(precision).flat, straddling, edge_density],
                  columns=["Precision", "$\lambda_s$", "$\lambda_e$"])
    df2 = pd.DataFrame(np.c_[np.asarray(success).flat, straddling, edge_density],
                  columns=["Success", "$\lambda_s$", "$\lambda_e$"])

    return (df1, df2)




def plotStraddelingEdgeOPE(wildcard="lambda_inner", savefilename='', format='pdf'):
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
    grid.map(plt.axhline, y=0.753958, ls=":", c=".5")

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

def getRunsForCVPR2016():
    runs=['SAMF', 'DSST', 'upd=3_hogANDhist_int_f1']
    runsNames =['lambda_SE_s0_e0.2', 'lambda_SE_s0_e0.5',
                'lambda_SE_s0.1_e0.3', 'lambda_SE_s0.2_e0.2', 'lambda_SE_s0.3_e0.4']
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


def plotOPE_SRE_TRE():
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
    #evaluator.experimentNames=corrected_names
    #saveFigureToFolder = '/Users/Ivan/Code/personal-website/Projects/Object_aware_tracking/images/multiScale/'
    saveFigureToFolder = '/Users/Ivan/Code/Tracking/Antrack/doc/technical_reports/images/'
    #saveFormat = ['png', 'pdf']
    saveFormat=['png']
    successAndPrecision = 'cvpr2016_SuccessAndPrecision_wu2013'
    histograms = 'cvpr2016_histogram_wu2013'


    for i in saveFormat:
        evaluator.evaluateFromSave(runs,successAndPrecisionPlotName=saveFigureToFolder+successAndPrecision+'.'+
                                                       i,histogramPlot=saveFigureToFolder+histograms+'.'+
                                                                                   i, alternativeNames=corrected_names)
        #evaluator.evaluateFromSave(runs,alternativeNames=corrected_names)

    #evaluator.evaluateFromSave(runs, alternativeNames=corrected_names)
def main():
    savefilename="/Users/Ivan/Code/Tracking/Antrack/doc/technical_reports/images/straddeling_edge_OPE"
    format='png'
    #plotStraddelingEdgeOPE(savefilename=savefilename,format=format)
    #plotOPE_SRE_TRE()
    plot_OPE_comparison()
if __name__ == "__main__":
    main()
