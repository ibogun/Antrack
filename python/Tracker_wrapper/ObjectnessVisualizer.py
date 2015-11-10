__author__ = 'Ivan'
import objectness_python
import tracker_python
from Dataset import VOT2015Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import gridspec
import re
import os
import time
import math
import copy

class ObjectnessVizualizer(object):
    """Class to perform objectness visualization"""

    def __init__(self, dataset, superpixels = 200, inner=0.9):
        """Constructor for ObjectnessVizualizer"""
        self.dataset = dataset
        self.superpixels = superpixels
        self.inner = inner


    @staticmethod
    def combinePlotsWithMean(full_image, H, img, mean, filename = None, axis_str = None):
        gs = gridspec.GridSpec(1, 3, width_ratios=[4, 2, 2])
        ax0 = plt.subplot(gs[0])
        ax0.imshow(full_image)
        ax0.axis('off')
        zvals = np.array(H)
        zvals2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #zvals = np.transpose(zvals)
        zvals = np.flipud(zvals)
        zvals2 = np.flipud(zvals2)
        cmap1 = plt.cm.jet
        cmap2 = plt.cm.gray
        cmap2._init() # create the _lut array, with rgba valuesHH

        alphas = np.linspace(0, 0.6, cmap2.N+3)
        cmap2._lut[:,-1] = alphas
        ax1 = plt.subplot(gs[1])
        ax1.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
        ax1.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')
        ax1.axis('off')
        if axis_str is not None:
            ax0.set_title(axis_str)

        ax1.set_title("Straddling")

        ax2=plt.subplot(gs[2])
        ax2.matshow(mean)
        ax2.axis('off')

        ax2.set_title("Mean")
        if filename is None:
            #plt.show()
            plt.draw()
            time.sleep(1)
        else:
            plt.savefig(filename,bbox_inches='tight',  dpi = 100)
            plt.close()
    @staticmethod
    def combinePlots(full_image, H, img,filename = None, axis_str = None):


        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        ax0.imshow(full_image)
        ax0.axis('off')
        zvals = np.array(H)

        #min_z = np.min(zvals.flatten(1))
        #max_z = np.max(zvals.flatten(1))
        #zvals = (zvals - min_z)/(max_z - min_z)
        zvals2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #zvals = np.transpose(zvals)
        zvals = np.flipud(zvals)
        zvals2 = np.flipud(zvals2)
        cmap1 = plt.cm.jet
        cmap2 = plt.cm.gray
        cmap2._init() # create the _lut array, with rgba valuesHH

        alphas = np.linspace(0, 0.6, cmap2.N+3)
        cmap2._lut[:,-1] = alphas
        ax1 = plt.subplot(gs[1])
        ax1.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
        ax1.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')
        ax1.axis('off')
        if axis_str is not None:
            ax0.set_title(axis_str)
        if filename is None:
            #plt.show()
            plt.draw()
            time.sleep(1)
        else:
            plt.savefig(filename,bbox_inches='tight',  dpi = 100)
            plt.close()

    @staticmethod
    def correctDims(box, width, height,  R):
        min_x = max(box[0]-R, 0)
        min_y = max(box[1]-R, 0)
        max_x = min(box[0]+R +box[2], width -1)
        max_y = min(box[1]+R+box[3], height -1)
        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def drawRectangle(image, box, R):
        n = image.shape[0]
        m = image.shape[1]

        c_x = n/2
        c_y = m/2
        pt1 = (max(c_y - R, box[2]/2), max(c_x - R, box[3]/2))
        pt2 = (min(c_y + R, m - box[2]/2), min(c_x + R, n - box[3]/2))

        cv2.rectangle(image, pt1, pt2, (0,255,100), 2)
        return image

    def evaluateImageAverageStraddling(self, video_number, frame_number = 0, saveFolder = None):
        video = self.dataset.video_folders[video_number]
        boxes = self.dataset.readGroundTruthAll(video)

        print video
        print len(boxes)
        images = self.dataset.getListOfImages(video)

        R = 60
        scale_R = 60
        min_size_half = 10
        min_scales=-15
        max_scales =8
        downsample=1.03
        shrink_one_size = 0

        s=re.split('/',video)
        video_name = s[len(s)-1]


        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        plt.show()

        i = frame_number


        obj = objectness_python.Objectness()
        box=boxes[i]
        im_name = images[i]
        img = cv2.imread(im_name,1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        height = img.shape[0]
        width = img.shape[1]

        (min_x, min_y, max_x, max_y) = self.correctDims(box, width, height, R)

        small_image = img[min_y:max_y, min_x :max_x]
        obj.readImage(im_name)


        obj.smallImage(R, box[0], box[1], box[2], box[3])

        a = obj.process(self.superpixels, self.inner, 0, R, scale_R, min_size_half, min_scales, max_scales,
                        downsample, shrink_one_size,
                        box[0], box[1], box[2], box[3])

        #obj.plot()
        c_x = box[0] - min_x + int(box[2]/2.0)
        c_y = box[1] - min_y + int(box[3]/2.0)
        counter = 1



        # reshuffle the list a bit
        a = a[1:min_scales] + [a[0]] + a[min_scales:len(a)]
        sums =np.zeros((len(a[0]),len(a[0][0])))
        counts = np.zeros((len(a[0]),len(a[0][0])))

        normalized=list()
        delay = 5
        for H,i in zip(a, range(0,len(a))):

            prevExists = (i-delay>=0)
            if (prevExists):
                objs_delay = np.array(a[i - delay])
            objs = np.array(H)
            mat = np.zeros((len(a[0]),len(a[0][0])))

            print np.max(H)
            for x in range(0,objs.shape[0]):
                for y in range(0, objs.shape[1]):

                    # get the new data
                    if objs[x,y]!=0:
                        counts[x,y]= counts[x,y]+1
                        sums[x,y] = sums[x,y] +objs[x,y]

                    # keep the moving average moving
                    if prevExists:
                        sums[x,y] = sums[x,y] - objs_delay[x,y]
                        if (objs_delay[x,y]!=0):
                            counts[x,y] = counts[x,y] -1

                    if counts[x,y]!= 0:
                        mat[x,y] = sums[x,y] / float(counts[x,y])

            normalized.append(mat)


        for H,h in zip(normalized,a):
            h=np.array(h)
            image_full = copy.deepcopy(img)
            small_image_copy = image_full[min_y:max_y, min_x :max_x]
            if ( counter == 1):
                half_width = box[2]/2.0
                half_height = box[3]/2.0
                width = box[2]
                height = box[3]
            else:
                half_width = ((box[2]/2)*math.pow(downsample, min_scales + counter - 1))
                half_height = ((box[3]/2)*math.pow(downsample, min_scales + counter - 1))

                width = int(half_width*2)
                height = int(half_height*2)


            pt1=(int(c_x - half_width), int(c_y - half_height))
            pt2=(int(c_x + half_width), int(c_y + half_height))
            cv2.rectangle(image_full, (pt1[0]+min_x, pt1[1]+min_y),(pt2[0]+min_x, pt2[1]+min_y), (100,0,150), 2)
            cv2.rectangle(image_full, (min_x, min_y), (max_x, max_y), (0,255,200),2)
            small_image_copy = self.drawRectangle(small_image_copy, (0,0,width, height) , R)

            print "processing image: ", " " , counter ,"/", len(a)
            if saveFolder is not None:
                directory = saveFolder + "/" + video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saveImage = directory+ str(1000 + counter) + ".png"

                if(os.path.isfile(saveImage)):
                    counter = counter + 1
                    continue
            else:
                saveImage = None

            axis_str = str(round(width/float(box[2])*100,2)) +"%"
            self.combinePlotsWithMean(image_full, H, small_image_copy,h,filename = saveImage, axis_str=axis_str)
            counter = counter + 1

        plt.close()

    def evaluateImage(self, video_number, frame_number = 0, saveFolder = None):
        video = self.dataset.video_folders[video_number]
        boxes = self.dataset.readGroundTruthAll(video)

        print video
        print len(boxes)
        images = self.dataset.getListOfImages(video)

        R = 60
        scale_R = 60
        min_size_half = 10
        min_scales=-15
        max_scales =8
        downsample=1.03
        shrink_one_size = 0

        s=re.split('/',video)
        video_name = s[len(s)-1]


        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        plt.show()

        i = frame_number


        obj = objectness_python.Objectness()
        box=boxes[i]
        im_name = images[i]
        img = cv2.imread(im_name,1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        height = img.shape[0]
        width = img.shape[1]

        (min_x, min_y, max_x, max_y) = self.correctDims(box, width, height, R)

        small_image = img[min_y:max_y, min_x :max_x]
        obj.readImage(im_name)


        obj.smallImage(R, box[0], box[1], box[2], box[3])

        a = obj.process(self.superpixels, self.inner, R, 0, scale_R, min_size_half, min_scales, max_scales,
                        downsample, shrink_one_size,
                        box[0], box[1], box[2], box[3])

        #obj.plot()
        c_x = box[0] - min_x + int(box[2]/2.0)
        c_y = box[1] - min_y + int(box[3]/2.0)
        counter = 1

        mean =np.zeros((len(a[0]),len(a[0][0])))
        counts = np.zeros((len(a[0]),len(a[0][0])))

        for H in a:
            objs = np.array(H)
            for x in range(0,objs.shape[0]):
                for y in range(0, objs.shape[1]):
                    if objs[x,y]!=0:
                        counts[x,y]= counts[x,y]+1
                        mean[x,y] = mean[x,y] +objs[x,y]


        for x in range(0,objs.shape[0]):
            for y in range(0, objs.shape[1]):
                if counts[x,y]!=0:
                   mean[x,y] = mean[x,y]/float(counts[x,y])

        for H in a:

            image_full = copy.deepcopy(img)
            small_image_copy = image_full[min_y:max_y, min_x :max_x]
            if ( counter == 1):
                half_width = box[2]/2.0
                half_height = box[3]/2.0
                width = box[2]
                height = box[3]
            else:
                half_width = ((box[2]/2)*math.pow(downsample, min_scales + counter - 1))
                half_height = ((box[3]/2)*math.pow(downsample, min_scales + counter - 1))

                width = int(half_width*2)
                height = int(half_height*2)


            pt1=(int(c_x - half_width), int(c_y - half_height))
            pt2=(int(c_x + half_width), int(c_y + half_height))
            cv2.rectangle(image_full, (pt1[0]+min_x, pt1[1]+min_y),(pt2[0]+min_x, pt2[1]+min_y), (100,0,150), 2)
            cv2.rectangle(image_full, (min_x, min_y), (max_x, max_y), (0,255,200),2)
            small_image_copy = self.drawRectangle(small_image_copy, (0,0,width, height) , R)

            print "processing image: ", " " , counter ,"/", len(a)
            if saveFolder is not None:
                directory = saveFolder + "/" + video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saveImage = directory+ str(1000 + counter) + ".png"

                if(os.path.isfile(saveImage)):
                    counter = counter + 1
                    continue
            else:
                saveImage = None

            axis_str = str(round(width/float(box[2])*100,2)) +"%"
            self.combinePlotsWithMean(image_full, H, small_image_copy, mean,filename = saveImage, axis_str=axis_str)
            counter = counter + 1

        plt.close()


    def evaluateDiscriminativeFunction(self, video_number, together=False, saveFolder=None):
        video = self.dataset.video_folders[video_number]
        boxes = self.dataset.readGroundTruthAll(video)

        print video
        print len(boxes)
        images = self.dataset.getListOfImages(video)
        bbox = boxes[0]

        R = 60
        scale_R = 60
        min_size_half = 10
        min_scales=0
        max_scales =0
        downsample=1.05
        shrink_one_size = 0

        s=re.split('/',video)
        video_name = s[len(s)-1]

        tracker = tracker_python.Antrack()
        tracker.initializeTracker()
        print images[0], bbox
        tracker.initialize(images[0], bbox[0], bbox[1], bbox[2], bbox[3])
        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        plt.show()
        for i in range(1, len(images)):

            print "processing image: ", " " , i ,"/", len(images)
            if saveFolder is not None:
                directory = saveFolder + "/" + video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saveImage = directory+ str(1000 + i) + ".png"

                if(os.path.isfile(saveImage)):
                    continue
            else:
                saveImage = None

            box=boxes[i]
            im_name = images[i]
            img = cv2.imread(im_name,1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height = img.shape[0]
            width = img.shape[1]

            (min_x, min_y, max_x, max_y) = self.correctDims(box, width, height, R)

            small_image = img[min_y:max_y, min_x :max_x]

            im_name = images[i]
            out = tracker.track(im_name)

            if i == 100:
                if together:
                    obj = objectness_python.Objectness()
                    obj.readImage(im_name)
                    obj.smallImage(R, box[0], box[1], box[2], box[3])
                    a_s = obj.processEdge(self.superpixels,self.inner, 0,
                                R, scale_R, min_size_half, min_scales, max_scales,
                                downsample, shrink_one_size,
                                box[0], box[1], box[2], box[3])

                    a_e = obj.processEdge(self.superpixels,self.inner, 0,
                                R, scale_R, min_size_half, min_scales, max_scales,
                                downsample, shrink_one_size,
                                box[0], box[1], box[2], box[3])

                    H_s=np.array(a_s[0])
                    H_e=np.array(a_e[0])

                a = tracker.calculateDiscriminativeFunction(im_name)
                H=np.array(a)
                H=H[min_x:max_x, min_y :max_y]
                H = np.transpose(H)

                if together:
                    min_z = np.min(H.flatten(1))
                    max_z = np.max(H.flatten(1))
                    H = (H - min_z)/(max_z - min_z)
                    H = H + 0.3* H_s + 0.3 * H_e

                print H.shape
                self.combinePlots(img, H, small_image, saveImage)

    def evaluateVideoEdge(self, video_number, saveFolder=None):
        video = self.dataset.video_folders[video_number]
        boxes = self.dataset.readGroundTruthAll(video)

        print video
        print len(boxes)
        images = self.dataset.getListOfImages(video)

        R = 60
        scale_R = 60
        min_size_half = 10
        min_scales=0
        max_scales =0
        downsample=1.05
        shrink_one_size = 0

        s=re.split('/',video)
        video_name = s[len(s)-1]


        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        plt.show()
        for i in range(0, len(images)):

            print "processing image: ", " " , i ,"/", len(images)
            if saveFolder is not None:
                directory = saveFolder + "/" + video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saveImage = directory+ str(1000 + i) + ".png"

                if(os.path.isfile(saveImage)):
                    continue
            else:
                saveImage = None

            obj = objectness_python.Objectness()
            box=boxes[i]
            im_name = images[i]
            img = cv2.imread(im_name,1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height = img.shape[0]
            width = img.shape[1]

            (min_x, min_y, max_x, max_y) = self.correctDims(box, width, height, R)

            small_image = img[min_y:max_y, min_x :max_x]
            obj.readImage(im_name)

            pt1=(box[0] - min_x, box[1] - min_y)
            pt2=(box[0] - min_x + box[2], box[1] -min_y + box[3])
            cv2.rectangle(small_image, pt1,pt2, (100,0,150), 2)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0,255,200),2)
            small_image = self.drawRectangle(small_image, box , R)
            obj.smallImage(R, box[0], box[1], box[2], box[3])

            a = obj.processEdge(self.superpixels,self.inner, 0,
                            R, scale_R, min_size_half, min_scales, max_scales,
                            downsample, shrink_one_size,
                            box[0], box[1], box[2], box[3])

            #obj.plot()
            H = a[0]

            print len(H), len(H[0])
            self.combinePlots(img, H, small_image, saveImage)

    def evaluateVideo(self, video_number, saveFolder=None):
        video = self.dataset.video_folders[video_number]
        boxes = self.dataset.readGroundTruthAll(video)

        print video
        print len(boxes)
        images = self.dataset.getListOfImages(video)

        R = 60
        scale_R = 60
        min_size_half = 10
        min_scales=0
        max_scales =0
        downsample=1.05
        shrink_one_size = 0

        s=re.split('/',video)
        video_name = s[len(s)-1]


        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        plt.show()
        for i in range(0, len(images)):

            print "processing image: ", " " , i ,"/", len(images)
            if saveFolder is not None:
                directory = saveFolder + "/" + video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saveImage = directory+ str(1000 + i) + ".png"

                if(os.path.isfile(saveImage)):
                    continue
            else:
                saveImage = None

            obj = objectness_python.Objectness()
            box=boxes[i]
            im_name = images[i]
            img = cv2.imread(im_name,1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height = img.shape[0]
            width = img.shape[1]

            (min_x, min_y, max_x, max_y) = self.correctDims(box, width, height, R)

            small_image = img[min_y:max_y, min_x :max_x]
            obj.readImage(im_name)

            pt1=(box[0] - min_x, box[1] - min_y)
            pt2=(box[0] - min_x + box[2], box[1] -min_y + box[3])
            cv2.rectangle(small_image, pt1,pt2, (100,0,150), 2)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0,255,200),2)
            #small_image = self.drawRectangle(small_image, box , R)
            obj.smallImage(R, box[0], box[1], box[2], box[3])

            a = obj.process(self.superpixels,self.inner, 0,
                            R, scale_R, min_size_half, min_scales, max_scales,
                            downsample, shrink_one_size,
                            box[0], box[1], box[2], box[3])

            #obj.plot()
            H = a[0]
            self.combinePlots(img, H, small_image, saveImage)


def straddlingInTime(save = False):
    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/vot2015/sequences'
    vot = VOT2015Dataset(root_folder)

    superpixels = 200

    obj = ObjectnessVizualizer(vot)

    #videos = [3, 30, 25]
    videos = [3]
    if save:
        saveOutputFolder =  '/Users/Ivan/Files/Results/Tracking/VOT2015_straddling_in_time'
    else:
        saveOutputFolder = None
    for v in videos:
        obj.evaluateVideo(v, saveOutputFolder)

def edgeDensityInTime(save = False):
    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/vot2015/sequences'
    vot = VOT2015Dataset(root_folder)


    obj = ObjectnessVizualizer(vot)

    #videos = [3, 30, 25]
    videos = [3]
    if save:
        saveOutputFolder =  '/Users/Ivan/Files/Results/Tracking/VOT2015_edgeDensity_in_time'
    else:
        saveOutputFolder = None
    for v in videos:
        obj.evaluateVideoEdge(v, saveOutputFolder)

def discriminativeFunctionInTime(together = True, save = False):
    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/vot2015/sequences'
    vot = VOT2015Dataset(root_folder)


    obj = ObjectnessVizualizer(vot)

    #videos = [3, 30, 25]
    videos = [3]
    if save:
        saveOutputFolder =  '/Users/Ivan/Files/Results/Tracking/VOT2015_discriminative_in_time'
    else:
        saveOutputFolder = None
    for v in videos:
        obj.evaluateDiscriminativeFunction(v,together=together, saveFolder=saveOutputFolder)

def straddlingInSpace( save = False):
    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/vot2015/sequences'
    vot = VOT2015Dataset(root_folder)

    superpixels = 200
    obj = ObjectnessVizualizer(vot, superpixels)

    videos = [3, 30, 25]
    #videos = [30]
    if save:
        saveOutputFolder =  '/Users/Ivan/Files/Results/Tracking/VOT2015_straddling_in_space'
    else:
        saveOutputFolder = None
    for v in videos:
        obj.evaluateImage(v, saveFolder=saveOutputFolder)

def straddelingAverageInSpace(save = False):
    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/vot2015/sequences'
    vot = VOT2015Dataset(root_folder)
    superpixels = 200
    obj = ObjectnessVizualizer(vot, superpixels)
    videos = [3, 30, 25]
    videos = [30]
    if save:
        saveOutputFolder =  '/Users/Ivan/Files/Results/Tracking/VOT2015_straddling_in_space_average'
    else:
        saveOutputFolder = None
    for v in videos:
        obj.evaluateImageAverageStraddling(v, saveFolder=saveOutputFolder)


if __name__ == "__main__":
    discriminativeFunctionInTime(together=True, save=True)
    #straddlingInTime(True)
    #edgeDensityInTime(False)