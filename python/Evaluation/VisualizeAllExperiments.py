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

from matplotlib.widgets import Slider, Button

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


    def precisionAndSuccessPlot(self,n=1000):

        d=dict()
        for experimentName,experimentData in self.experiments.data.iteritems():

            e=VisualizeExperiment(self.dataset,experimentData)

            (x_pr, y_pr, x_s, y_s)=e.precisionAndSuccessDataAveragedPerVideo(n)

            d[experimentName]= (x_pr, y_pr, x_s, y_s)


        # do the plotting business here


        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = len(x_pr)

        titleFontSize = 16;
        headerFontSize = 14;
        axisFontSize = 12;
        lineWidth = 1.8;

        legendSize = 9;

        plt.figure(figsize=(13,9))

        with plt.style.context('grayscale'):
            i = 2

            idx=1
            for expName in d.keys():
                (x_pr, y_pr, x_s, y_s)=d[expName]
                handlesLegendPrecision = list()
                handlesLegendSuccess = list()
                p = np.trapz(y_pr, x=x_pr) / 50
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


def main(argv=None):
    if argv is None:
        argv = sys.argv

    wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"

    datasetType = 'wu2013'
    experimentType='default'

    runName = './Runs/obj_raw_linear_pre0_filter0_edge0_straddling0_prior0.p'

    run = loadPickle(runName)

    # print run

    vidName = 'basketball'

    dataset = Dataset(wu2013GroundTruth, datasetType)

    viz = VisualizeAllExperiments(dataset, run)

    viz.precisionAndSuccessPlot()
    # exp1=run.data['default']
    #
    # vizOne=VisualizeExperiment(dataset,exp1)
    #
    # vizOne.precisionAndSuccessPlot(vidName)

    #viz.show(vidName,experimentRunNumber=0,experiment='default')
    #viz.show(vidName,100)

    #viz.barplot()
    #viz.precisionAndSuccessPlot(vidName)
    #viz.show(vidName)
    # vidData=[x for x in dataset.dictData if x['name']==vidName][0]


if __name__ == "__main__":
    main()