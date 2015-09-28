__author__ = 'Ivan'
import abc
import os
import cv2
import numpy as np
import glob
class Dataset(object):
    """Tracking dataset class"""
    __metaclass__ = abc.ABCMeta
    def __init__(self, root_folder =  None, image_format = 'jpg'):
        """Constructor for Dataset"""
        self.root_folder = root_folder
        self.image_format = image_format

        if root_folder is not None:
            self.listVideos()

    def getListOfImages(self, video):
        return glob.glob(video +"/*." +self.image_format)

    @staticmethod
    def showImageAndBox(I, box, color=(225,65,105)):
        cv2.rectangle(I, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), color, thickness=2)
        cv2.imshow("Tracking box", I)
        cv2.waitKey()

    @abc.abstractmethod
    def readGroundTruthOneLine(self, image_path, line_counter = 0):
        raise NotImplementedError("Implementation of the readGroundTruthOneLine() depends on the "
                                  "Dataset, so implement it!")

    @abc.abstractmethod
    def readGroundTruthAll(self, image_path):
        raise NotImplementedError("Implementation of the readGroundTruthAll() depends on the "
                                  "Dataset, so implement it!")

    def listVideos(self):
        sub_folders = os.listdir(self.root_folder)
        sub_folders = [self.root_folder + "/" + x for x in os.listdir(self.root_folder) if ((not x.startswith('.'))
                                                                                  and (os.path.isdir(self.root_folder + "/" + x)))]

        self.video_folders = sub_folders


class VOT2015Dataset(Dataset):
    """VOT 2015 Dataset class"""

    def boundingBoxFromCoordinates(self, x_coordinates, y_coordinates):

        # find average center
        c_x = 0
        c_y = 0

        for x,y in zip(x_coordinates, y_coordinates):
            c_x = c_x + x
            c_y = c_y + y

        c_x = c_x / len(x_coordinates)
        c_y = c_y / len(y_coordinates)

        # sort x and y and subtract from the largest two the smallest two
        x_sorted = sorted(x_coordinates)
        y_sorted = sorted(y_coordinates)

        width = ((x_sorted[2] + x_sorted[3]) - (x_sorted[1] + x_sorted[0])) / 2.0
        height = ((y_sorted[2] + y_sorted[3]) - (y_sorted[1] + y_sorted[0])) / 2.0

        top_left_x = c_x - width / 2.0
        top_left_y = c_y - height / 2.0

        return (int(top_left_x), int(top_left_y), int(width), int(height))


    def readGroundTruthOneLine(self, video_folder, line_counter = 0):

        ground_truth_file = video_folder + "/groundtruth.txt"

        f = open(ground_truth_file, 'r')


        counter = 0
        for line in f:
            if counter == line_counter:
                line = line.rstrip()
                numbers = line.split(",")
                record = [float(x) for x in numbers]
                x_coordinates = record[::2]
                y_coordinates = record[1::2]
                break
            counter = counter + 1

        return self.boundingBoxFromCoordinates(x_coordinates, y_coordinates)

    def readGroundTruthAll(self, video_folder):
        '''
        Return ALL bounding boxes from the ground truth
        :param video_folder: folder with the ground truth file
        :return: list of bounding boxes where each bounding box is a (top_left_x, top_left_y, width, height)
        '''
        ground_truth_file = video_folder + "/groundtruth.txt"

        f = open(ground_truth_file, 'r')

        rectangles = list()

        counter = 0
        for line in f:
            line = line.rstrip()
            numbers = line.split(",")
            record = [float(x) for x in numbers]
            x_coordinates = record[::2]
            y_coordinates = record[1::2]
            rectangles.append(self.boundingBoxFromCoordinates(x_coordinates, y_coordinates))

        return rectangles


if __name__ == '__main__':

    root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/antrack/sequences'
    vot = VOT2015Dataset(root_folder)

    video = vot.video_folders[0]

    for video in vot.video_folders:
        if  not video.endswith("graduate"):
            continue
        rectangle = vot.readGroundTruthOneLine(video)
        images = vot.getListOfImages(video)
        I = cv2.imread(images[0])
        Dataset.showImageAndBox(I, rectangle)

    print "Total number of videos: ", len(vot.video_folders)




