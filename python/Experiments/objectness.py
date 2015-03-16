import sys
import csv
sys.path.append('../modules')
sys.path.append('../Evaluation')
import objectness
from  Evaluation import  DatasetEvaluation

#
# a=objectness.Objectness()
#
# imnamge="/Users/Ivan/Files/Data/Tracking_benchmark/basketball/img/0001.jpg"
# a.getObjectness(imnamge,198,214,34,81)
#
# print a.getStraddling
# print a.getEdgeness

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

    dict_straddling = dict()
    dict_edgeness = dict()
    for d in dataset.dictData:

        # if d["name"]!="walking":
        #     continue


        print d["name"]

        #
        # print len(d["boxes"])
        # print len(d["images"])
    #d=dataset.dictData[0]



        l_straddling=list()
        l_edgeness = list()
        #idx = 0

        minIdx=0;

        for idx in range(minIdx,min(len(d["boxes"]),len(d["images"]))):

            #print idx;

            imName=d["images"][idx];
            #box=d["boxes"]

            # if idx>0:
            #     break;

            bbox= d["boxes"][idx]


            # if idx %(len(d["images"]))/(10.0) ==0:
            #     print ".",
            #print bbox;

            # skip small bounding boxes
            # if bbox[2]<15 or bbox[3]<15:
            #     continue;
            #bbox[0]=bbox[0]-1
            #bbox[1]=bbox[1]-1
            #print bbox, bbox[0]+bbox[2], bbox[1]+bbox[3]

            objness.getObjectness(imName,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            l_straddling.append(objness.getStraddling)
            l_edgeness.append(objness.getEdgeness)

            #idx=idx+1

            objness.plot();

        dict_straddling[d["name"]]=l_straddling
        dict_edgeness[d["name"]] =l_edgeness

    #saveDictionaryAsCSV(dict_straddling,"straddling.csv")
    #saveDictionaryAsCSV(dict_edgeness,"edgeness.csv")

    # print dataset.dictData[0]["name"]
    # print dataset.dictData[0]["images"]
    # print dataset.dictData[0]["boxes"]
