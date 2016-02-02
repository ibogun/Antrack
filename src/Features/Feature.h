//
//  Feature.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef Robust_Struck_Feature_h
#define Robust_Struck_Feature_h

#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "armadillo"

class Feature {

    
public:

    virtual ~Feature(){}
    
    virtual cv::Mat prepareImage(cv::Mat* imageIn)=0;
    
    //virtual int calculateDimension()=0;
    
   
    virtual arma::mat calculateFeature(cv::Mat& processedImage,std::vector<cv::Rect>& rects)=0;
    virtual int calculateFeatureDimension()=0;
    
    virtual std::string getInfo()=0;

    virtual void setParams(const std::unordered_map<std::string, std::string> & map) {
    }

    arma::mat reshapeYs(std::vector<cv::Rect>& locations){
        
        arma::mat y((int)locations.size(),5,arma::fill::zeros);
        arma::rowvec tmp(5,arma::fill::zeros);
        
        // for every location
        for (int l=0; l<locations.size(); ++l) {
            tmp<<l<<locations[l].x<<locations[l].y<<locations[l].width<<locations[l].height<<arma::endr;
            y.row(l)=tmp;
        }
        
        return y;
    }
};

#endif
