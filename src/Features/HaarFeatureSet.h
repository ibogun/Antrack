//
//  HaarFeatureSet.h
//  STR
//
//  Created by Ivan Bogun on 7/4/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __STR__HaarFeatureSet__
#define __STR__HaarFeatureSet__

#include <iostream>
#include "HaarFeature.h"
#include <vector>

//typedef cv::Mat mat;

class HaarFeatureSet{
    
private:
    
    cv::Mat image;                  // image where locations are to be sampled
    arma::rowvec scales;
    int featureSize;
    int normalization;

    
public:
    
    HaarFeatureSet(const cv::Mat&,arma::rowvec&,int,int);
    
    void calculateFeatures(std::vector<cv::Rect>&,std::vector<cv::Rect>&,arma::mat&,arma::mat&);
   
    std::string getInfo(){
        std::string r="Haar of size "+std::to_string(featureSize);
        return r;
    };
    
};

#endif /* defined(__STR__HaarFeatureSet__) */
