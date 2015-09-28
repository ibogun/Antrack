import  objectness_python
from Dataset import VOT2015Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import gridspec
import  re
def combinePlots(full_image, H, img, filename = None):

    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.imshow(full_image)
    ax0.axis('off')
    zvals = np.array(H)
    zvals2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    zvals = np.transpose(zvals)
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
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')

def correctDims(box,width, height, R):
    min_x = max(box[0]-R, 0)
    min_y = max(box[1]-R, 0)
    max_x = min(box[0]+R +box[2], width -1)
    max_y = min(box[1]+R+box[3], height -1)
    return (min_x, min_y, max_x, max_y)

def drawRectangle(image, box, R):
    n = image.shape[0]
    m = image.shape[1]
    
    c_x = n/2
    c_y = m/2
    pt1 = (max(c_y - R, box[2]/2), max(c_x - R, box[3]/2))
    pt2 = (min(c_y + R, m - box[2]/2), min(c_x + R, n - box[3]/2))

    #pt2 = (c_y + R,c_x + R)
    #pt1 = (c_y - R,c_x - R)
    cv2.rectangle(image, pt1, pt2, (0,255,100), 2)
    return image

root_folder = '/Users/Ivan/Code/Tracking/Antrack/matlab/vot-toolkit/antrack/sequences'
vot = VOT2015Dataset(root_folder)

video = vot.video_folders[3]

boxes = vot.readGroundTruthAll(video)

print video
print len(boxes)
images = vot.getListOfImages(video)

R = 60
scale_R = 60
min_size_half = 10
min_scales=0
max_scales =0
downsample=1.05
shrink_one_size = 0


s=re.split('/',video)
print s[len(s)-1]

for i in range(0, 1):
    obj = objectness_python.Objectness()
    print "processing image: ", " " , i ,"/", len(images)
    box=boxes[i]
    im_name = images[i]
    img = cv2.imread(im_name,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height = img.shape[0]
    width = img.shape[1]
    
    (min_x, min_y, max_x, max_y) = correctDims(box, width, height, R)

    small_image = img[min_y:max_y, min_x :max_x]
    obj.readImage(im_name)
    
    pt1=(box[0] - min_x, box[1] - min_y)
    pt2=(box[0] - min_x + box[2], box[1] -min_y + box[3])
    cv2.rectangle(small_image, pt1,pt2, (100,0,150), 2)
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0,255,200),2)
    small_image = drawRectangle(small_image, box , R)
    obj.smallImage(R, box[0], box[1], box[2], box[3])

    a = obj.process(R, scale_R, min_size_half, min_scales, max_scales, downsample, shrink_one_size,
                    box[0], box[1], box[2], box[3])
    
    
    H = a[0]

    #combinePlots(img, H, small_image)
