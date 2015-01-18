//
//  Struck.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Struck__
#define __Robust_Struck__Struck__

#include <stdio.h>
#include "OLaRank_old.h"
#include "../Features/Feature.h"
#include "LocationSampler.h"
#include <vector>
#include "../Datasets/DataSetWu2013.h"
#include "../Datasets/Dataset.h"
#include <unordered_map>
#include "../Filter/KalmanFilterGenerator.h"


class Struck {
    
    OLaRank_old* olarank;
    Feature* feature;
    LocationSampler* samplerForUpdate;
    LocationSampler* samplerForSearch;
    
    KalmanFilter_my filter;
    
    std::vector<cv::Rect> boundingBoxes;
    cv::Rect lastLocation;
    cv::Rect lastRectFilterAndDetectorAgreedOn;
    
    int display;
    
    std::unordered_map<int,cv::Mat> frames;
    
    // for plotting support vectors (used only if display==2)
    cv::Mat canvas;
    
    int gridSearch=2;
    int R=30;
    int framesTracked=0;
    
    bool useFilter;
    bool updateTracker;
    
    int seed=1;
    
    void copyFromRectangleToImage(cv::Mat& canvas,cv::Mat& image,cv::Rect rect,int step,cv::Vec3b color);

    
public:
    
    Struck();
    
    Struck(OLaRank_old* olarank_,Feature* feature_,LocationSampler* samplerSearch_,LocationSampler* samplerUpdate_,bool useFilter_,int display_){
        olarank = olarank_;
        feature = feature_;
        samplerForSearch=samplerSearch_;
        samplerForUpdate = samplerUpdate_;
        useFilter=useFilter_;
        display = display_;};
    
    void setParameters(OLaRank_old* olarank_,Feature* feature_,LocationSampler* samplerSearch_,LocationSampler* samplerUpdate_,bool useFilter_,int display_){
      
        this->olarank=olarank_;
        this->feature=feature_;
        this->samplerForSearch=samplerSearch_;
        this->samplerForUpdate=samplerUpdate_;
        this->useFilter=useFilter_;
        this->display=display_;
    };
    
    
    std::vector<cv::Rect> getBoundingBoxes(){
        return this->boundingBoxes;
    };
    
    void videoCapture();
    
    void initialize(cv::Mat& image,cv::Rect& location);
    
    void allocateCanvas(cv::Mat&);
    
    void track(cv::Mat& image);
    
    void updateDebugImage(cv::Mat* canvas,cv::Mat& img, cv::Rect &bestLocation,cv::Scalar colorOfBox);
    
    void applyTrackerOnDataset(Dataset* dataset,std::string rootFolder, int videoNumber=-1);
    
    void applyTrackerOnVideoWithinRange(Dataset* dataset,std::string rootFolder,int videoNumber, int frameFrom, int frameTo);
    
    void saveResults(string fileName);
};

#endif /* defined(__Robust_Struck__Struck__) */
