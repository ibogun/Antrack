//
//  DatasetVOT2014.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/22/15.
//
//

#ifndef __Robust_tracking_by_detection__DatasetVOT2014__
#define __Robust_tracking_by_detection__DatasetVOT2014__

#include <stdio.h>
#include "Dataset.h"
#include <unordered_map>

#include "armadillo"


class DatasetVOT2014:public Dataset {


public:

    //hashtable vidToIndexHashTable;
    std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder);
    std::vector<cv::RotatedRect> readComplete(std::string);
    std::vector<cv::Rect> readGroundTruth(std::string);


    static cv::RotatedRect constructRotatedRect(std::vector<float> v);


    static float findAngle(cv::Point2f[]);


    void showVideo(std::string rootFolder, int vidNumber);

    std::string getInfo(){
        std::stringstream ss;

        ss<<"VOT 2014 dataset (ECCV 2014)\n";
        ss<<"25 videos\n";
        return ss.str();
    }

    ~DatasetVOT2014(){}
};

#endif /* defined(__Robust_tracking_by_detection__DatasetVOT2014__) */
