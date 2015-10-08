__author__ = 'Ivan'
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button

from DatasetEvaluation import Dataset,loadPickle,Evaluator


class VisualizeExperiment(object):
    """VisualizeExperiment the object here and there"""

    def __init__(self, dataset,run):
        """Constructor for VisualizeExperiment"""
        self.dataset=dataset
        self.run=run
        # what for do we need the dataset?

    def show(self,vidName,experimentType='SRE',experimentRunNumber=0,delay=1):
        """Show the movie

        Args:
            self,vidName

        Returns:
            None
        """


        movie=self.getMovie(vidName, experimentType, experimentRunNumber)

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.subplot(gs[0])
        l=plt.imshow(movie[0])
        plt.grid(b=None)
        plt.axis("off")
        axcolor = 'lightgoldenrodyellow'
        axframe = plt.axes([0.25, 0.2, 0.45, 0.03], axisbg=axcolor)

        resetax = plt.axes([0.5-0.03, 0.25, 0.05, 0.03])
        buttonPlay = Button(resetax, 'Play/Stop', color=axcolor, hovercolor='0.975')

        plt.subplot(gs[1])

        (x_pr, y_pr, x_s, y_s) = self.precisionAndSuccessPlotData(vidName, experimentType, n=1000)

        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = 2

        titleFontSize = 16;
        headerFontSize = 14;
        axisFontSize = 12;
        lineWidth = 1.8;

        legendSize = 9;

        with plt.style.context('grayscale'):
            i = 1
            handlesLegend = list()
            p = np.trapz(y_pr, x=x_pr) / 50
            s = np.trapz(y_s, x=x_s)

            x_pr= x_pr/50
            p = np.ma.round(p, 2)
            s = np.ma.round(s, 2)

            color = cm(1. * i / NUM_COLORS)
            red_patch = mpatches.Patch(label='precision' + ' [' + str(p) + ']',
                                       color=cm(1. * 2 / NUM_COLORS))
            blue_path = mpatches.Patch(label='success' + ' [' + str(s) + ']',
                                       color=cm(1. * 1 / NUM_COLORS))
            handlesLegend.append(red_patch)
            handlesLegend.append(blue_path)



            plt.plot(x_s, y_s, linewidth=lineWidth, color=cm(1. * 1 / NUM_COLORS))

            plt.ylim([0, 1.1])
            plt.xlim([-0.02, 1.1])

            plt.grid(b=False)
            # plt.subplot(1, 2, 2)
            #
            plt.plot(x_pr, y_pr, linewidth=lineWidth, color=cm(1. * 2 / NUM_COLORS))

            plt.title("Evaluation", fontsize=headerFontSize)
            plt.grid(b=False)
            plt.xlabel('threshold', fontsize=axisFontSize)
            plt.ylabel('metric', fontsize=axisFontSize)

            plt.legend(handles=handlesLegend, prop={'size': legendSize})


        minIdx=0
        maxIdx=len(movie)-1
        sframe = Slider(axframe, 'Frame', minIdx, maxIdx, valinit=0)

        speed_frame = plt.axes([0.5-0.05, 0.15, 0.1, 0.03], axisbg=axcolor)
        speed_slider = Slider(speed_frame, 'Speed', 1, 50, valinit=1)

        def update(val):
            frame = np.round(sframe.val)
            frame = max(frame, minIdx)
            frame = min(frame, maxIdx)

            frame = frame - minIdx
            l.set_data(movie[int(frame)])


        sframe.on_changed(update)





        class PlayStopButton(object):
            ind = 0
            pauseInterval = 1/25.0
            def play(self, event):
                self.ind += 1
                for i in range(int(sframe.val), maxIdx):

                    sframe.set_val(i)
                    plt.pause(self.pauseInterval)

                    if self.ind % 2 == 0:
                        return

            def updateSpeed(self,event):

                self.pauseInterval= 1.0/max(1,speed_slider.val);
        callback = PlayStopButton()
        speed_slider.on_changed(callback.updateSpeed)
        # def play(event):
        #     # starting from sframe.val
        #
        #     #playTimesClicked= playTimesClicked+1
        #     for i in range(int(sframe.val),maxIdx):
        #         l.set_data(movie[int(i)])
        #         plt.pause(pauseInterval)
        #
        #         if playTimesClicked%2==0:
        #             return

        buttonPlay.on_clicked(callback.play)


        plt.show()
        # for frame in movie:
        #
        #     cv2.imshow("tracking results",frame)
        #     cv2.waitKey(delay)
        #
        # cv2.destroyAllWindows()


    def precisionAndSuccessPlotData(self, vidName, experimentType,experimentNumber=0,n=1000):
        """Get the data necessary for plotting precision and recall

        Args:
            vidName,n=1000

        Returns:
            (x_pr, y_pr, x_s, y_s)
        """

        gt_data = [x for x in self.dataset.data if x[0] == vidName][0]

        #tracker_data = [x for x in self.run.data[experimentType].data if x[0] == vidName][0]
        print vidName
        tracker_data = [x for x in self.run.data if x[0] == vidName][0]
        (x_pr, y_pr, x_s, y_s) = Evaluator.evaluateSingleVideo(tracker_data, gt_data,
                                                               experimentNumber=0)

        for index in range(1,len(tracker_data[1])):
            (x_pr1, y_pr1, x_s1, y_s1) = Evaluator.evaluateSingleVideo(tracker_data, gt_data,
                                                                   experimentNumber=index)
            x_pr= x_pr+x_pr1
            y_pr=y_pr+y_pr1
            x_s=x_s+x_s1
            y_s=y_s+y_s1


        x_pr=x_pr/float(len(tracker_data[1]))
        y_pr = y_pr / float(len(tracker_data[1]))
        x_s = x_s / float(len(tracker_data[1]))
        y_s = y_s / float(len(tracker_data[1]))


        return (x_pr, y_pr, x_s, y_s)

    def precisionAndSuccessDataAveragedPerRun(self,videoName,n=1000):
        (x_pr, y_pr, x_s, y_s) = self.precisionAndSuccessPlotData(videoName, 0, n=1000)
        for experimentRunNumber in range(1, len(self.run.data[0])):
            # another loop -> for every experiment run on the video


            (x_pr_next, y_pr_next, x_s_next, y_s_next) = self.precisionAndSuccessPlotData(videoName, experimentRunNumber,
                                                                                       n=1000)

            x_pr = x_pr + x_pr_next
            y_pr = y_pr + y_pr_next
            x_s = x_s + x_s_next
            y_s = y_s + y_s_next

        x_pr = x_pr / len(self.run.data[0])
        y_pr = y_pr / len(self.run.data[0])
        x_s = x_s / len(self.run.data[0])
        y_s = y_s / len(self.run.data[0])

        return (x_pr,y_pr,x_s,y_s)


    def precisionAndSuccessDataAveragedPerVideo(self,n=1000):

        vidNames = [x[0] for x in self.dataset.data];

        (x_pr, y_pr, x_s, y_s) = self.precisionAndSuccessDataAveragedPerRun(vidNames[0])
        for vidIdx in range(1, len(vidNames)):
            vidName = vidNames[vidIdx];
            (x_pr_next, y_pr_next, x_s_next, y_s_next) = self.precisionAndSuccessDataAveragedPerRun(vidName)

            x_pr = x_pr + x_pr_next
            y_pr = y_pr + y_pr_next
            x_s = x_s + x_s_next
            y_s = y_s + y_s_next

        l=len(vidNames)

        x_pr = x_pr / l
        y_pr = y_pr / l
        x_s = x_s / l
        y_s = y_s / l


        return (x_pr,y_pr,x_s,y_s)


    def precisionAndSuccessPlot(self,vidName,n=1000):
        """Plot precision and success plots for a single video run

        Args:
            n=1000

        Returns:
            None
        """

        # change dataste and runs

        #evaluator = Evaluator(dataset, runs)

        (x_pr, y_pr, x_s, y_s)= self.precisionAndSuccessPlotData(vidName,n=1000)

        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = len(x_pr)

        titleFontSize=16;
        headerFontSize = 14;
        axisFontSize = 12;
        lineWidth = 1.8;

        legendSize = 9;

        with plt.style.context('grayscale'):

            i=4
            handlesLegendPrecision = list()
            handlesLegendSuccess = list()
            p = np.trapz(y_pr, x=x_pr) / 50
            s = np.trapz(y_s, x=x_s)

            p = np.ma.round(p, 2)
            s = np.ma.round(s, 2)

            color = cm(1. * i / NUM_COLORS)
            red_patch = mpatches.Patch(label=self.run.trackerLabel + ' [' + str(p) + ']',
                                       color=color)
            blue_path = mpatches.Patch(label=self.run.trackerLabel + ' [' + str(s) + ']',
                                       color=color)
            handlesLegendPrecision.append(red_patch)
            handlesLegendSuccess.append(blue_path)

            plt.suptitle(vidName, fontsize=titleFontSize)
            plt.subplot(1, 2, 1)

            plt.plot(x_s, y_s, linewidth=lineWidth, color=cm(1. * i / NUM_COLORS))
            plt.title('success', fontsize=headerFontSize)

            plt.ylim([0, 1.1])
            plt.xlim([-0.02, 1.1])
            plt.xlabel('Overlap threshold', fontsize=axisFontSize)
            plt.ylabel('Success rate', fontsize=axisFontSize)

            plt.legend(handles=handlesLegendSuccess, prop={'size': legendSize})
            plt.grid(b=False)
            plt.subplot(1, 2, 2)

            plt.plot(x_pr, y_pr, linewidth=lineWidth, color=cm(1. * i / NUM_COLORS))
            plt.ylim([0, 1.1])
            plt.xlim([-0.5, 51])
            plt.title("precision", fontsize=headerFontSize)
            plt.grid(b=False)
            plt.xlabel('Location error threshold', fontsize=axisFontSize)
            plt.ylabel('Precision', fontsize=axisFontSize)

            plt.legend(handles=handlesLegendPrecision, prop={'size': legendSize}, loc=2)

            plt.show()



    def barplot(self, n=1000):
        """Plots barplot with precision and success for specific run
        
        Args:
            n=1000
           
        Returns:
            nothing
        """

        precision=list()
        success=list()

        names=list()
        n_groups = len(self.dataset.data)

        for tracker_data in self.run.data:
            gt_data = [x for x in self.dataset.data if x[0] == tracker_data[0]][0]

        #for gt_data,tracker_data in zip(self.dataset.data,self.run.data):

            if gt_data[0]!=tracker_data[0]:

                print "Should be happening"
                return

            names.append(gt_data[0])
            (x_pr, y_pr, x_s, y_s) = Evaluator.evaluateSingleVideo(tracker_data, gt_data)

            p = np.trapz(y_pr, x=x_pr) / 50
            s = np.trapz(y_s, x=x_s)

            p = np.ma.round(p, 2)
            s = np.ma.round(s, 2)

            precision.append(p)
            success.append(s)


        rotation=90

        xTicksFontSize=12;

        index = np.arange(n_groups)
        plt.figure(figsize=(15,10))
        plt.suptitle(self.run.trackerLabel,fontsize=xTicksFontSize+6)
        plt.subplots_adjust(bottom=0.2)
        plt.subplot(1, 2, 1)


        idx_success = [i[0] for i in sorted(enumerate(success), key=lambda x: x[1])]
        idx_precision = [i[0] for i in sorted(enumerate(precision), key=lambda x: x[1])]

        successTrackerNames = [names[x] for x in idx_success]
        sorted_success = [success[x] for x in idx_success]

        precisionTrackerNames = [names[x] for x in idx_precision]
        sorted_precision = [precision[x] for x in idx_precision]

        plt.xticks(index, successTrackerNames, rotation=rotation, fontsize=xTicksFontSize)
        plt.bar(index, sorted_success, align="center")
        plt.ylim((0, 1))

        plt.yticks(fontsize=xTicksFontSize)
        mean_success=np.round(sum(success)/(1.0*len(success)),2)

        plt.title("Success "+"["+str(mean_success)+"]",fontsize=xTicksFontSize+4)
        plt.subplot(1, 2, 2)

        plt.bar(index, sorted_precision, align="center")
        plt.xticks(index, precisionTrackerNames, rotation=rotation,fontsize=xTicksFontSize)
        plt.ylim((0, 1))
        plt.yticks(fontsize=xTicksFontSize)
        mean_precision = np.round(sum(sorted_precision) / (1.0 * len(sorted_precision)), 2)
        plt.title("Precision " + "[" + str(mean_precision) + "]", fontsize=xTicksFontSize + 4)


        plt.show()

    def getMovie(self,vidName, experimentType,experimentRunNumber=0):
        """ Creates a movie based on the tracking results

        Args:
            vidName - name of the video to generate frames

        Returns:
            movie - list of frames with bounding boxes
        """


        color_GT=(0,0,255)
        color_tracker=(255,0,0)

        thickness_gt=2
        thickness_tracker=2

        vidData_gt = [x for x in self.dataset.dictData if x['name'] == vidName][0]


        runData=self.run.data[experimentType]

        vidData_tracker= [x for x in runData.data if x[0]==vidName][0]

        def getPointsFromRectangle(rect):

            pt1=(int(rect[0]),int(rect[1]))
            pt2=(int(rect[0]+rect[2]),int(rect[1]+rect[3]))
            return (pt1,pt2)

        boxes_gt=vidData_gt['boxes']

        #boxes_tracker=vidData_tracker[1][experimentRunNumber]


        if experimentType=='SRE':
            colors= [[198, 151, 103],
                     [181, 98, 208],
                     [114, 209, 87],
                     [156, 165, 202],
                     [141, 206, 175],
                     [69, 78, 79],
                     [199, 86, 133],
                     [209, 82, 57],
                     [205, 196, 69],
                     [91, 69, 127],
                     [86, 112, 53],
                     [106, 49, 43]];
        else:
            colors= [[59, 43, 47],
                     [114, 217, 85],
                     [202, 81, 205],
                     [217, 133, 50],
                     [106, 172, 201],
                     [192, 210, 145],
                     [202, 137, 125],
                     [213, 73, 134],
                     [201, 150, 200],
                     [208, 68, 59],
                     [207, 214, 70],
                     [86, 139, 58],
                     [117, 109, 203],
                     [78, 84, 123],
                     [103, 213, 175],
                     [117, 61, 35],
                     [77, 99, 71],
                     [123, 52, 96],
                     [171, 147, 66],
                     [192, 197, 192]]

        movie=list()
        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS=len(vidData_tracker[1])

        for image,idx in zip(vidData_gt['images'],range(0,len(vidData_gt['images']))):

            I= cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)

            (pt1,pt2)=getPointsFromRectangle(boxes_gt[idx])
            cv2.rectangle(I,pt1,pt2,color_GT,thickness=thickness_gt)

            for boxes_tracker,index in zip(vidData_tracker[1],range(0,len(vidData_tracker[1]))):
                (pt1, pt2) = getPointsFromRectangle(boxes_tracker[idx])
                cv2.rectangle(I, pt1, pt2, tuple(colors[index]),thickness=thickness_tracker)


            movie.append(I)

        return movie

def main(argv=None):
    if argv is None:
        argv = sys.argv


    wu2013GroundTruth = "/Users/Ivan/Files/Data/wu2013"

    datasetType = 'wu2013'



    runName = './Runs/lambda=0.1_hogANDhist_int_f1.p'

    run=loadPickle(runName)

    #print run

    experimenType='default'
    run=run.data[experimenType]
    #
    vidName='jogging-2'

    dataset = Dataset(wu2013GroundTruth, datasetType)


    viz=VisualizeExperiment(dataset,run)

    #viz.show(vidName, experimenType)

    #viz.barplot()
    viz.precisionAndSuccessPlot(vidName)
    #viz.show(vidName)
    # vidData=[x for x in dataset.dictData if x['name']==vidName][0]




if __name__ == "__main__":
    main()