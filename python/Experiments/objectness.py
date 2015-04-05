import sys
import csv
sys.path.append('../modules')
sys.path.append('../Evaluation')
import objectness
from  Evaluation import DatasetEvaluation
import cv2
import math
import numpy as np


def saveDictionaryAsCSV(d,path):
    w = csv.writer(open(path, "w"))
    for key, val in d.items():
        w.writerow([key, val])


if __name__ == "__main__":
    wu2013results="/Users/Ivan/Files/Results/Tracking/wu2013"
    wu2013GroundTruth="/Users/Ivan/Files/Data/Tracking_benchmark"

    vot2014Results="/Users/Ivan/Files/Results/Tracking/vot2014"
    vot2014GrounTruth="/Users/Ivan/Files/Data/vot2014"

    datasetType='wu2013'

    dataset=DatasetEvaluation.Dataset(wu2013GroundTruth,datasetType)

    objness=objectness.Objectness()


    # dictionary which maps video_name -> dict, which in its turn maps
    # experiment_name -> array of measurements
    dict_straddling = dict()
    dict_edgeness = dict()

    edge_t1=1
    edge_t2=200;
    inner_rectangle=0.95
    nSuperpixels=200;

    number_to_skip=0
    for d in dataset.dictData:

        # if d["name"]="bolt":
        #     continue

        # if number_to_skip<=15:
        #     number_to_skip=number_to_skip+1
        #     continue;


        print d["name"]


        pts_per_experiment=8 # number of points on the circle

        num_experiments=5    # number of experiments

        experiment_names=np.linspace(0,1,num=num_experiments)

        experiment_names=np.delete(experiment_names,0)
        experiment_names=np.round(experiment_names,decimals=2)

        l_straddling=list()
        l_edgeness = list()

        minIdx=0;

        d_edge_video=dict()
        d_straddling_video=dict()


        list_of_experiments_edge=list()
        list_of_experiments_straddling=list()

        for i in range(0,num_experiments):
            list_of_experiments_edge.append(list())
            list_of_experiments_straddling.append(list())

        for idx in range(minIdx, min(len(d["boxes"]), 3)):
        #for idx in range(minIdx,min(len(d["boxes"]),len(d["images"]))):

            print idx,;

            imName=d["images"][idx];

            bbox= d["boxes"][idx]

            objness.readImage(imName)
            objness.initializeStraddling(nSuperpixels,inner_rectangle)
            objness.initializeEdgeDensity(edge_t1,edge_t2,inner_rectangle)

            #objness.initialize(imName,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            l_straddling.append(objness.getStraddling(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            l_edgeness.append(objness.getEdgeness(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

            image=cv2.imread(imName)

            maxDistance=math.sqrt(math.pow(bbox[2]/2,2)+ math.pow(bbox[3] / 2, 2))

            distances=np.linspace(0, maxDistance,num=num_experiments)
            distances=np.delete(distances,0)

            n = image.shape[0]
            m = image.shape[1]

            rho_values=np.linspace(0,2*np.pi,pts_per_experiment)

            mid_x=bbox[0]+bbox[2]/2.0
            mid_y=bbox[1]+bbox[3]/2.0

            for i in range(0,num_experiments-1):
                # perform experiment and add averaged results to list_of_experiments[i].append(average result)
                dist = distances[i]

                edgeness=0
                straddling=0

                number_good_boxes=0

                for index in range(0,len(rho_values)):

                    rho=rho_values[index]


                    new_center_x=mid_x+math.sin(rho)*dist
                    new_center_y=mid_y+math.cos(rho)*dist

                    experiment_box=[new_center_x- bbox[2] / 2.0, new_center_y - bbox[3] / 2.0]

                    if experiment_box[0]>=0 and experiment_box[1]>=0 and experiment_box[0]+bbox[2]<m \
                            and experiment_box[1]+bbox[3]<n:

                        #objness.initialize(imName, int(experiment_box[0]), int(experiment_box[1]), int(bbox[2]), int(bbox[3]))




                        number_good_boxes= number_good_boxes+1

                        edgeness = edgeness + objness.getEdgeness(int(experiment_box[0]), int(experiment_box[1]),
                                                                  int(bbox[2]), int(bbox[3]))
                        straddling = straddling +objness.getStraddling(int(experiment_box[0]), int(experiment_box[1]),
                                                                       int(bbox[2]), int(bbox[3]))

                        #objness.plot()

                if number_good_boxes>0:
                    edgeness = float(edgeness)/number_good_boxes
                    straddling = float(straddling) / number_good_boxes

                else:
                    edgeness=np.inf
                    straddling=np.inf

                list_of_experiments_edge[i].append(edgeness)
                list_of_experiments_straddling[i].append(straddling)
                        #
                    # check if box is whithing the image


            #objness.plot();
        d_edge_video['origin']=l_edgeness
        d_straddling_video['origin']=l_straddling

        for experiment_idx in range(0,len(experiment_names)):
            d_edge_video[str(experiment_names[experiment_idx])]=list_of_experiments_edge[experiment_idx]
            d_straddling_video[str(experiment_names[experiment_idx])] = list_of_experiments_straddling[experiment_idx]


        # here add all of the experiments

        # for each experiment add to dictionaries above
        print " "
        dict_straddling[d["name"]]= d_straddling_video # this has to be dictionary
        dict_edgeness[d["name"]] = d_edge_video    # this too

        number_to_skip= number_to_skip+1

    #saveDictionaryAsCSV(dict_straddling,"straddling.csv")
    #saveDictionaryAsCSV(dict_edgeness,"edgeness.csv")
