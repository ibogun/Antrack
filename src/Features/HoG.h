//
//  HoG.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/26/15.
//
//

#ifndef __Robust_tracking_by_detection__HoG__
#define __Robust_tracking_by_detection__HoG__

#include <stdio.h>
#include "Feature.h"

class HoG:public  Feature{
    
    cv::Size size;
    
    cv::HOGDescriptor* d;
public:
    HoG(cv::Size size_);
    
    cv::Mat prepareImage(cv::Mat* imageIn);
    
    arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
    
    int calculateFeatureDimension();
    
    std::string getInfo();

    ~HoG(){}
    
};

#endif /* defined(__Robust_tracking_by_detection__HoG__) */
