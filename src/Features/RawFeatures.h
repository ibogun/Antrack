//
//  RawFeatures.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__RawFeatures__
#define __Robust_Struck__RawFeatures__

#include <stdio.h>
#include "Feature.h"

class RawFeatures:public Feature {
    
    
public:
    int size;
    RawFeatures(int size_){this->size=size_;};
    
    cv::Mat prepareImage(cv::Mat* imageIn);
    arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
    int calculateFeatureDimension(){return size*size;};
    
    std::string getInfo(){
        std::string r="Raw features with size: "+std::to_string(size)+" \n";
        return r;
    };


    ~RawFeatures(){
    }
};

#endif /* defined(__Robust_Struck__RawFeatures__) */
