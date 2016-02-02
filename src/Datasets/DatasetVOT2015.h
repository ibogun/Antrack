//
//  DatasetVOT2015.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/22/15.
//
//

#ifndef __Robust_tracking_by_detection__DatasetVOT2015__
#define __Robust_tracking_by_detection__DatasetVOT2015__

#include <stdio.h>
#include "DatasetVOT2014.h"
#include <unordered_map>

#include "armadillo"


class DatasetVOT2015:public DatasetVOT2014 {


public:

    //hashtable vidToIndexHashTable;
    /*
    std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder);
    std::vector<cv::RotatedRect> readComplete(std::string);
    std::vector<cv::Rect> readGroundTruth(std::string);


    static cv::RotatedRect constructRotatedRect(std::vector<float> v);
    */
    static cv::Rect constructRectangle(std::vector<float> v);
    std::vector<cv::Rect> readGroundTruth(std::string);

    std::vector<std::vector<float>> readAllRecords(std::string);

    /*
    static float findAngle(cv::Point2f[]);


    void showVideo(std::string rootFolder, int vidNumber);
    */
    std::string getInfo(){
        std::stringstream ss;

        ss<<"VOT 2015 dataset (ICCV 2015)\n";
        ss<<"60 videos\n";
        return ss.str();
    }

    ~DatasetVOT2015(){}
};

#endif /* defined(__Robust_tracking_by_detection__DatasetVOT2015__) */
