__author__ = 'Ivan'
# Import C++ implementation of the objectness.
import sys
import math
sys.path.append('../../Experiments')
import objectness

sys.path.append('../../Evaluation')
# from  Evaluation import  DatasetEvaluation
import DatasetEvaluation
import cv2
from matplotlib import pyplot as plt
import numpy as np

import copy

### CHANGE THIS PARAMETERS, IF NECESSARY
vidName = "dudek"
imageNumber = 150;

fig_x_size = 13
fig_y_size = 10

wu2013GroundTruth = "/Users/Ivan/Files/Data/Tracking_benchmark"
datasetType = 'wu2013'

dataset = DatasetEvaluation.Dataset(wu2013GroundTruth, datasetType)
d = dataset.dictData;



# get the dictionary
vidDictionary = [x for x in d if x["name"] == vidName][0]
# get the bounding box
box = vidDictionary['boxes'][imageNumber]
# get the image
ImName = vidDictionary['images'][imageNumber]
I = cv2.imread(ImName)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

pt1 = (int(box[0]), int(box[1]))
pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
cv2.rectangle(I, pt1, pt2, (0, 0, 255), 2)

box = np.round(box)


# create objectness object
objness = objectness.Objectness()

edge_t1 = 1
edge_t2 = 200
inner_rectangle = 0.95
nSuperpixels = 200

# needs around 1272 seconds per objectness measure
import os.path
import time

outfileEdge = vidName + '_edge_' + str(imageNumber) + '_3D';
outfileStraddling = vidName + '_straddling_' + str(imageNumber) + '_3D';

radius = 20;
minScaleLevel = -5;
maxScaleLevel = 10;
scaleChange = 1.05;

thirdDim = maxScaleLevel - minScaleLevel + 1;

if (not os.path.isfile(outfileEdge + '.npy')) or (not os.path.isfile(outfileStraddling + '.npy')):
    O_edge = np.zeros((I.shape[0], I.shape[1], thirdDim))
    O_straddling = np.zeros((I.shape[0], I.shape[1], thirdDim))
    t0 = time.time()
    print I.shape
    # iterate over all possible locations of the boundig box

    for scale in range(minScaleLevel, maxScaleLevel+1):
        print scale,
        width = box[2] * np.power(scaleChange, scale);
        height = box[3] * np.power(scaleChange, scale);


        objness.readImage(ImName);
        objness.initializeStraddling(nSuperpixels, inner_rectangle);
        objness.initializeEdgeDensity(edge_t1, edge_t2, inner_rectangle);

        for y in range(0, I.shape[0]):
            #print y, ;
            for x in range(0, I.shape[1]):
                # create a bounding box centered at x,y
                b = [x - width / 2, y - height / 2, width, height]

                # make sure the bounding box fits the image
                if (b[0] < 0 or b[0] + b[2] >= I.shape[1]) or (b[1] < 0 or b[1] + b[3] >= I.shape[0]):
                    continue

                # objness.initialize(ImName, int(b[0]), int(b[1]), int(b[2]), int(b[3]))

                if (not os.path.isfile(outfileEdge + '.npy')):
                    O_edge[y][x][scale- minScaleLevel] = objness.getEdgeness(int(b[0]), int(b[1]), int(b[2]), int(b[3]))

                if (not os.path.isfile(outfileStraddling + '.npy')):
                    O_straddling[y][x][scale- minScaleLevel] = objness.getStraddling(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    t1 = time.time()

    total_time = t1 - t0;
    np.save(outfileEdge, O_edge)
    np.save(outfileStraddling, O_straddling)
else:
    O_edge = np.load(outfileEdge + '.npy')
    O_straddling = np.load(outfileStraddling + '.npy')

#print t1-t0," total time"
from matplotlib.widgets import Slider, Button, RadioButtons




import numpy
import pylab
from matplotlib.widgets import Slider

#data = O_straddling  # 3d-array with 100 frames 256x256
data= O_edge

for scale in range(minScaleLevel, maxScaleLevel + 1):

    s=np.sum(O_edge[:,:,scale-minScaleLevel])
    s=np.sum(s)
    O_edge[:,:,scale-minScaleLevel]= O_edge[:, :, scale - minScaleLevel]/s

    s = np.sum(O_straddling[:, :, scale - minScaleLevel])
    s = np.sum(s)
    O_straddling[:, :, scale - minScaleLevel] = O_straddling[:, :, scale - minScaleLevel] / s


O_combined= numpy.multiply(O_edge,O_straddling)

findMin = lambda x: np.min(np.min(x))
findMax = lambda x: np.max(np.max(x))

frame = 0


fig = plt.figure(figsize=(15,10))
ax1 = pylab.subplot(2,2,1)

image= cv2.cvtColor(cv2.imread(ImName), cv2.COLOR_BGR2RGB)

pt1 = (int(box[0]), int(box[1]))
pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))

I = copy.deepcopy(image)

cv2.rectangle(I, pt1, pt2, (0, 255, 0), thickness=2)

ll=pylab.imshow(I)
pylab.grid(b=None)
plt.axis("off")

ax = pylab.subplot(2, 2, 2)
pylab.subplots_adjust(left=0.25, bottom=0.25)

l_combined = pylab.imshow(O_combined[:, :, frame], cmap=plt.get_cmap("jet"))  # shows 256x256 image, i.e. 0th frame
l_combined.set_clim(vmin=findMin(O_combined[:, :, 0]), vmax=findMax(O_combined[:, :, 0]))
# fig.colorbar(l)
plt.grid(b=False)
plt.axis("off")

ax = pylab.subplot(2,2, 3)
pylab.subplots_adjust(left=0.25, bottom=0.25)

l_edge = pylab.imshow(O_edge[:, :, frame], cmap=plt.get_cmap("jet"))  # shows 256x256 image, i.e. 0th frame
l_edge.set_clim(vmin=findMin(O_edge[:,:,0]), vmax=findMax(O_edge[:,:,0]))
#fig.colorbar(l)
plt.grid(b=False)
plt.axis("off")

ax = pylab.subplot(2,2,4)
l_straddling = pylab.imshow(O_straddling[:, :, frame], cmap=plt.get_cmap("jet"))  # shows 256x256 image, i.e. 0th frame
l_straddling.set_clim(vmin=findMin(O_straddling[:, :, 0]), vmax=findMax(O_straddling[:, :, 0]))
# fig.colorbar(l)
plt.grid(b=False)
plt.axis("off")

axcolor = 'lightgoldenrodyellow'
axframe = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sframe = Slider(axframe, 'Scale',
                minScaleLevel, maxScaleLevel, valinit=0)


def update(val):
    frame = numpy.around(sframe.val)

    width = box[2] * np.power(scaleChange, frame);
    height = box[3] * np.power(scaleChange, frame);

    rect = [box[0]+box[2]/2 - width / 2, box[1]+box[3]/2 - height / 2, width, height]

    frame=max(frame,minScaleLevel)
    frame=min(frame,maxScaleLevel - minScaleLevel)



    pt1 = (int(rect[0]), int(rect[1]))
    pt2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))

    I=copy.deepcopy(image)

    cv2.rectangle(I,pt1,pt2,(0,255,0),thickness=2)

    ll.set_data(I)

    frame=frame-minScaleLevel
    l_edge.set_data(O_edge[:, :, frame])
    l_edge.set_clim(vmin=findMin(O_edge[:, :, frame]), vmax=findMax(O_edge[:, :, frame]))

    l_straddling.set_data(O_straddling[:, :, frame])
    l_straddling.set_clim(vmin=findMin(O_straddling[:, :, frame]), vmax=findMax(O_straddling[:, :, frame]))


    l_combined.set_data(O_combined[:, :, frame])
    l_combined.set_clim(vmin=findMin(O_combined[:, :, frame]), vmax=findMax(O_combined[:, :, frame]))



print numpy.linalg.norm(O_combined-O_edge)
sframe.on_changed(update)
pylab.show()