//
//  MultiFeature.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/8/15.
//
//

#ifndef __Robust_tracking_by_detection__MultiFeature__
#define __Robust_tracking_by_detection__MultiFeature__

#include <stdio.h>
#include "Feature.h"
#include <vector>
class MultiFeature:public Feature{
    
    std::vector<Feature*> features;
    
public:
    
    MultiFeature(std::vector<Feature*>& fs){
        this->features=fs;
    };
    
    cv::Mat prepareImage(cv::Mat* imageIn);
    
    //virtual int calculateDimension()=0;
    
    
    arma::mat calculateFeature(cv::Mat& processedImage,std::vector<cv::Rect>& rects);
    int calculateFeatureDimension();
    
    std::string getInfo();
};

#endif /* defined(__Robust_tracking_by_detection__MultiFeature__) */
