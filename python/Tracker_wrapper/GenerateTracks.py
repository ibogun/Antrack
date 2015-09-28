__author__ = 'Ivan'

import struck_tracker
import cv2
import matplotlib.pylab as plt
import glob
import numpy as np
import copy
import os

def showImage(I):
    cv2.imshow("",I)
    cv2.waitKey()

def showMatrix(I):
    plt.matshow(I, fignum=100, cmap=plt.cm.gray)

    plt.show()

def showImageAndBox(I, box, color=(225,65,105)):
    cv2.rectangle(I, (box[0], box[1]), (box[2] + box[0], box[3]+box[1]), color, thickness=2)
    cv2.imshow("Tracking box", I)
    cv2.waitKey(100)

if __name__ == '__main__':
    tracker = struck_tracker.Antrack()

    all_dirs = [ 'coffee']

    for k in all_dirs:

        folder = "/Users/Ivan/Downloads/kod/" + k +"/"

        images = folder+ "images/"
        labels = folder+'labels/'

        image_files = glob.glob(images+ "image00*.png")
        label_files = glob.glob(labels+"*.png")
        cropped_files_folder = folder+"/cropped/"

        c = 0
        label = []
        trackers = []
        add_pixels = 5

        if not os.path.exists(cropped_files_folder):
            os.makedirs(cropped_files_folder)

        # for every label file
        for label_file in label_files:
            image_name = image_files[c]

            if label_file == '/Users/Ivan/Downloads/kod/coffee/labels/image000240.png':
                break

            label = cv2.imread(label_file, 0)
            max_id = label.max()
            # for every label
            for j in range(1, max_id + 1):
                mask = copy.deepcopy(label)
                t = struck_tracker.Antrack()
                t.initializeTracker()

                (n, m) = mask.shape
                mask[mask != j] = 0

                idx_x, idx_y = np.nonzero(mask)
                if (len(idx_x) == 0) or (len(idx_y) == 0):
                    continue


                cropped_files_folder_category = cropped_files_folder + "/" + str(j)
                if os.path.isfile(cropped_files_folder_category + "/" + str(c) + ".png"):
                    continue

                if j == 8:
                    if abs(idx_y[0] - idx_y[len(idx_y)-1])> 100:
                        idx_y = idx_y[idx_y < 600]

                min_x = max(idx_x.min() - add_pixels, 0)
                max_x = min(idx_x.max() + add_pixels, n - 1)
                min_y = max(idx_y.min() - add_pixels, 0)
                max_y = min(idx_y.max() + add_pixels, m - 1)

                width = max_x - min_x
                height = max_y - min_y
                boxes = list()
                box = [min_y, min_x, height, width]
                # showImageAndBox(I, box)
                t.initialize(image_name, box[0], box[1], box[2], box[3])
                boxes.append(box)
                print "Current label: ", j, " in ", k, " frames: ", c, "-", c + 30
                # track for 30 frames
                for frame_id in range(c, min(c+30, len(image_files))):
                    box = t.track(image_files[frame_id])
                    I = cv2.imread(image_files[frame_id])
                    print frame_id,

                    #showImageAndBox(I, (box))

                    boxes.append(box)

                print ""
                if not os.path.exists(cropped_files_folder_category):
                    os.makedirs(cropped_files_folder_category)

                for idx in range(0, 30):
                    if (c + idx) >= len(image_files):
                        break
                    I = cv2.imread(image_files[c + idx])
                    box = boxes[idx]
                    cropped = I[box[1]:(box[3] + box[1]), box[0]:(box[2] + box[0])]
                    cv2.imwrite(cropped_files_folder_category+"/"+str(c + idx)+".png", cropped)


            c = c+30
