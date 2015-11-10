__author__ = 'Ivan'
import sys
from DatasetEvaluation import Dataset, loadPickle, Evaluator
from generatePythonFilePickle import AllExperiments

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


class VisualizeAllExperiments(object):
    """VisualizeExperiment the object here and there"""

    def __init__(self, dataset, experiments):
        """Constructor for VisualizeExperiment"""
        self.dataset = dataset

        self.experiments= experiments
        # what for do we need the dataset?

    def show(self, vidName,experimentRunNumber=0, experiment='default', delay=1):
        """Show the movie

        Args:
            self,vidName

        Returns:
            None
        """


        if experiment=='default':
            idx=0;
        elif experiment=='TRE':
            idx=1
        elif experiment=='SRE':
            idx=2
        else:
            raise NameError('Experiment: '+experiment+' does not exist, try one of these: default, TRE, SRE.')


        experiment= VisualizeExperiment(self.dataset,self.experiments.data[experiment])

        experiment.show(vidName,experimentRunNumber=experimentRunNumber,delay=delay)


    def precisionAndSuccessPlot(self,n=1000, all=True):

        d=dict()
        for experimentName,experimentData in self.experiments.data.iteritems():

            e=VisualizeExperiment(self.dataset,experimentData)

            (x_pr, y_pr, x_s, y_s)=e.precisionAndSuccessDataAveragedPerVideo(n)

            d[experimentName]= (x_pr, y_pr, x_s, y_s)

            if not all:
                break;


        # do the plotting business here


        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = len(x_pr)

        titleFontSize = 16;
        headerFontSize = 14;
        axisFontSize = 12;
        lineWidth = 1.8;

        legendSize = 9;

        plt.figure(figsize=(13,9),)

        with plt.style.context('grayscale'):
            i = 2

            idx=1
            for expName in d.keys():
                if (not all) and (expName != 'default'):
                    break
                (x_pr, y_pr, x_s, y_s)=d[expName]
                handlesLegendPrecision = list()
                handlesLegendSuccess = list()
                p = np.trapz(y_pr, x=x_pr) / 51
                s = np.trapz(y_s, x=x_s)

                p = np.ma.round(p, 2)
                s = np.ma.round(s, 2)

                color = cm(1. * i / NUM_COLORS)

                if idx==1:
                    red_patch = mpatches.Patch(label=self.experiments.data[expName].trackerLabel+' [' + str(p) + ']',
                                               color=color)
                    #self.run.trackerLabel
                else:
                    red_patch = mpatches.Patch(label=' [' + str(p) + ']',
                                               color=color)
                blue_path = mpatches.Patch(label=' [' + str(s) + ']',
                                           color=color)
                handlesLegendPrecision.append(red_patch)
                handlesLegendSuccess.append(blue_path)

                #plt.suptitle(expName, fontsize=titleFontSize)
                plt.subplot(3, 2, idx)




                plt.plot(x_s, y_s, linewidth=lineWidth, color=cm(1. * i / NUM_COLORS))

                if idx==1:
                    plt.title('success', fontsize=headerFontSize)

                plt.ylim([0, 1.1])
                plt.xlim([-0.02, 1.1])

                if idx == 5:
                    plt.xlabel('Overlap threshold', fontsize=axisFontSize)


                plt.ylabel('Success rate', fontsize=axisFontSize)



                idx = idx + 1
                plt.legend(handles=handlesLegendSuccess, prop={'size': legendSize})
                plt.grid(b=False)
                plt.subplot(3, 2, idx)



                plt.plot(x_pr, y_pr, linewidth=lineWidth, color=cm(1. * i / NUM_COLORS))
                plt.ylim([0, 1.1])
                plt.xlim([-0.5, 51])

                if idx==2:
                    plt.title("precision", fontsize=headerFontSize)
                plt.grid(b=False)

                if idx==6:
                    plt.xlabel('Location error threshold', fontsize=axisFontSize)


                plt.ylabel('Precision', fontsize=axisFontSize)

                ax2 = plt.twinx()
                ax2.set_ylabel(expName, color='black')
                ax2.grid(b=False)
                idx = idx + 1
                plt.legend(handles=handlesLegendPrecision, prop={'size': legendSize}, loc=2)

            plt.show()

    def barplotDefault(self, n = 1000, savefile='',sort=False):
        plt.figure(figsize=(13,9))

        rotation = 90

        xTicksFontSize = 10;

        index=1;
        import seaborn as sn


        mean_success=0
        mean_precision=0

        sList=list()
        pList=list()


        expName = 'default'
        experiment = self.experiments.data[expName]
        print expName

        precision = list()
        success = list()
        names=list()


        for videoData in experiment.data:

            gt=[x for x in self.dataset.data if x[0]==videoData[0]][0]


            p=0
            s=0
            names.append(videoData[0])

            for expRunIndex in range(0,len(videoData[1])):
                (x_pr, y_pr, x_s, y_s)=Evaluator.evaluateSingleVideo(videoData,gt,experimentNumber=expRunIndex, n=n)

                p1= np.ma.round(np.trapz(y_pr, x=x_pr) / 51, 2)
                s1= np.ma.round(np.trapz(y_s, x=x_s), 2)

                p =p+ p1
                s =s+ s1

                sList.append(s1)
                pList.append(p1)




            p=p/(float(len(videoData[1])))
            s = s/ (float(len(videoData[1])))

            precision.append(p)
            success.append(s)

            #break

        bothMetrics=[x+y for x,y in zip(success,precision)]
        # barplot precision
        n_groups = len(self.dataset.data)

        indexPlot = np.arange(n_groups)

        if index == 1:
            ax1 = plt.subplot(1, 2, index)
        else:
            plt.subplot(1, 2, index)

        #
        if sort:
            idx_sorted = [i[0] for i in sorted(enumerate(bothMetrics), key=lambda x: x[1])]
        else:
            idx_sorted = [i[0] for i in enumerate(bothMetrics)]
        successTrackerNames = [names[x] for x in idx_sorted]
        sorted_success = [success[x] for x in idx_sorted]

        precisionTrackerNames = [names[x] for x in idx_sorted]
        sorted_precision = [precision[x] for x in idx_sorted]

        plt.xticks(indexPlot, successTrackerNames, rotation=rotation, fontsize=xTicksFontSize)

        plt.bar(indexPlot, sorted_success, align="center")

        plt.ylim((0, 1))

        plt.yticks(fontsize=xTicksFontSize)

        mean_success = mean_success+ np.round(sum(success) / (1.0 * len(success)), 3)
        mean_precision=mean_precision+ np.round(sum(precision) / (1.0 * len(precision)), 3)
        #plt.title("Success " + "[" + str(mean_success) + "]", fontsize=xTicksFontSize + 4)

        index = index + 1;
        plt.subplot(1, 2, index)

        plt.xticks(indexPlot, precisionTrackerNames, rotation=rotation, fontsize=xTicksFontSize)
        plt.bar(indexPlot, sorted_precision, align="center")
        plt.ylim((0, 1))

        plt.yticks(fontsize=xTicksFontSize)
        if index == 2:
            ax2 = plt.subplot(1, 2, index)


        ax3 = plt.twinx()
        ax3.set_ylabel(expName, color='black',fontsize=xTicksFontSize+4)
        ax3.grid(b=False)


        sFinal=np.round(sum(sList) / (float(len(sList))),3)
        pFinal = np.round(sum(pList) / (float(len(pList))), 3)

        ax1.set_title(
            "Success " + "[" + str(sFinal) + "] / " + self.experiments.data['default'].trackerLabel,
            fontsize=xTicksFontSize + 4)
        ax2.set_title("Precision " + "[" + str(pFinal) + "] / " + self.experiments.data[
            'default'].trackerLabel, fontsize=xTicksFontSize + 4)

        # ax1.set_title("Success " + "[" + str(mean_success/2.0) + "] / "+ self.experiments.data['default'].trackerLabel, fontsize=xTicksFontSize + 4)
        # ax2.set_title("Precision " + "[" + str(mean_precision / 2.0) + "] / "+ self.experiments.data[
        #     'default'].trackerLabel, fontsize=xTicksFontSize + 4)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if savefile=='':
            plt.show()
        else:
            plt.savefig(savefile)
            plt.close()




    def barplot(self,n=1000,savefile=''):


        plt.figure(figsize=(13,9))

        rotation = 90

        xTicksFontSize = 10;

        index=1;
        import seaborn as sn


        mean_success=0
        mean_precision=0

        sList=list()
        pList=list()

        for expName,experiment in self.experiments.data.iteritems():


            print expName

            precision = list()
            success = list()
            names=list()


            for videoData in experiment.data:

                gt=[x for x in self.dataset.data if x[0]==videoData[0]][0]


                p=0
                s=0
                names.append(videoData[0])

                for expRunIndex in range(0,len(videoData[1])):
                    (x_pr, y_pr, x_s, y_s)=Evaluator.evaluateSingleVideo(videoData,gt,experimentNumber=expRunIndex, n=n)

                    p1= np.ma.round(np.trapz(y_pr, x=x_pr) / 51, 2)
                    s1= np.ma.round(np.trapz(y_s, x=x_s), 2)

                    p =p+ p1
                    s =s+ s1

                    if expName!='default':
                        sList.append(s1)
                        pList.append(p1)




                p=p/(float(len(videoData[1])))
                s = s/ (float(len(videoData[1])))

                precision.append(p)
                success.append(s)

                #break

            bothMetrics=[x+y for x,y in zip(success,precision)]
            # barplot precision
            n_groups = len(self.dataset.data)

            indexPlot = np.arange(n_groups)

            if index == 1:
                ax1 = plt.subplot(3, 2, index)
            else:
                plt.subplot(3, 2, index)

            idx_sorted = [i[0] for i in sorted(enumerate(bothMetrics), key=lambda x: x[1])]

            successTrackerNames = [names[x] for x in idx_sorted]
            sorted_success = [success[x] for x in idx_sorted]

            precisionTrackerNames = [names[x] for x in idx_sorted]
            sorted_precision = [precision[x] for x in idx_sorted]

            plt.xticks(indexPlot, successTrackerNames, rotation=rotation, fontsize=xTicksFontSize)

            plt.bar(indexPlot, sorted_success, align="center")

            plt.ylim((0, 1))

            plt.yticks(fontsize=xTicksFontSize)

            if expName!='default':
                mean_success = mean_success+ np.round(sum(success) / (1.0 * len(success)), 2)
                mean_precision=mean_precision+ np.round(sum(precision) / (1.0 * len(precision)), 2)
            #plt.title("Success " + "[" + str(mean_success) + "]", fontsize=xTicksFontSize + 4)

            index = index + 1;
            plt.subplot(3, 2, index)

            plt.xticks(indexPlot, precisionTrackerNames, rotation=rotation, fontsize=xTicksFontSize)
            plt.bar(indexPlot, sorted_precision, align="center")
            plt.ylim((0, 1))

            plt.yticks(fontsize=xTicksFontSize)
            if index == 2:
                ax2 = plt.subplot(3, 2, index)
            else:
                plt.subplot(3, 2, index)

            ax3 = plt.twinx()
            ax3.set_ylabel(expName, color='black',fontsize=xTicksFontSize+4)
            ax3.grid(b=False)


            # barplot success
            index=index+1;


        sFinal=np.round(sum(sList) / (float(len(sList))),2)
        pFinal = np.round(sum(pList) / (float(len(pList))), 2)

        ax1.set_title(
            "Success " + "[" + str(sFinal) + "] / " + self.experiments.data['default'].trackerLabel,
            fontsize=xTicksFontSize + 4)
        ax2.set_title("Precision " + "[" + str(pFinal) + "] / " + self.experiments.data[
            'default'].trackerLabel, fontsize=xTicksFontSize + 4)

        # ax1.set_title("Success " + "[" + str(mean_success/2.0) + "] / "+ self.experiments.data['default'].trackerLabel, fontsize=xTicksFontSize + 4)
        # ax2.set_title("Precision " + "[" + str(mean_precision / 2.0) + "] / "+ self.experiments.data[
        #     'default'].trackerLabel, fontsize=xTicksFontSize + 4)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if savefile=='':
            plt.show()
        else:
            plt.savefig(savefile)







def main(argv=None):
    if argv is None:
        argv = sys.argv

    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"

    datasetType = 'wu2013'
    experimentType='default'

    runName = './Runs/a28_hist_int_f1.p'

    run = loadPickle(runName)

    # print run

    vidName = 'basketball'

    dataset = Dataset(wu2013GroundTruth, datasetType)

    viz = VisualizeAllExperiments(dataset, run)

    #viz.precisionAndSuccessPlot()
    # exp1=run.data['default']
    #
    # vizOne=VisualizeExperiment(dataset,exp1)
    #
    # vizOne.precisionAndSuccessPlot(vidName)

    #viz.show(vidName,experimentRunNumber=0,experiment='default')
    #viz.show(vidName,100)


    runsNames = glob.glob('./Runs/*.p')

    for r in runsNames:
        print r
        #viz.barplot()
    #viz.precisionAndSuccessPlot(vidName)
    #viz.show(vidName)
    # vidData=[x for x in dataset.dictData if x['name']==vidName][0]

def generateAllVizualiations():

    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    folderForGraphs='./Visualizations/'
    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)
    runsNames = glob.glob('./Runs/upd=3*.p')
    runsNames =['lambda_nonorm_s0_e0.5', 'lambda_nonorm_s0.2_e0.2',
                'lambda_nonorm_s0.3_e0.4', 'lambda_nonorm_s0.4_e0.4', 'lambda_nonorm_s0.5_e0.3']


    #runsNames =['lambda_SE_s0_e0.2', 'lambda_SE_s0_e0.5',
    #            'lambda_SE_s0.1_e0.3', 'lambda_SE_s0.2_e0.2', 'lambda_SE_s0.3_e0.4']

    for i in range(0,len(runsNames)):
        runsNames[i]="./Runs/"+runsNames[i]+".p"
    formatSave='pdf'

    regexp= re.compile("(.*\/)(.+)(.p)")


    for runName in runsNames:



        m=re.match(regexp,runName)

        name=m.group(2)
        print name
        run = loadPickle(runName)
        viz = VisualizeAllExperiments(dataset, run)
        viz.barplot(savefile=folderForGraphs+name+"."+ formatSave)

def generateDefaultVizualiations(wildcard):

    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    folderForGraphs='./Visualizations/'
    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)
    runsNames = glob.glob("./Runs/" + wildcard + "*.p")
    #runsNames = glob.glob('./Runs/upd=3_hogANDhist_int*.p')
    formatSave='pdf'

    regexp= re.compile("(.*\/)(.+)(.p)")


    for runName in runsNames:
        m=re.match(regexp,runName)

        name=m.group(2)
        print name
        run = loadPickle(runName)

        run.data['TRE'].data=[]
        run.data['SRE'].data=[]
        viz = VisualizeAllExperiments(dataset, run)
        viz.barplotDefault(savefile=folderForGraphs+name+"."+ formatSave)
        #viz.precisionAndSuccessPlot(all=False)

def compareDefaultPlots(wildcard="lambda_SE"):
    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"
    datasetType = 'wu2013'
    dataset = Dataset(wu2013GroundTruth, datasetType)
    runsNames = glob.glob("./Runs/" + wildcard + "*.p")

    formatSave='pdf'

    regexp= re.compile("(.*\/)(.+)(.p)")
    #s0_e0.2
    #s0_e0.5
    #s0.1_e0.3
    #s0.2_e0.2
    #s0.3_e0.4
    d=dict()
    runs=list()
    for runName in runsNames:
        m=re.match(regexp,runName)
        name=m.group(2)
        print name
        run = loadPickle(runName)
        run.trackerLabel=runName
        run.data['TRE'].data=[]
        run.data['SRE'].data=[]
        d[runName] = run
        runs.append(run)
    evaluator = Evaluator(dataset, runs)
    evaluator.evaluateSingleTracker(runs[0])


if __name__ == "__main__":
    #generateAllVizualiations()
    generateDefaultVizualiations("TGPR")
    #compareDefaultPlots()