//
//  HoGandRawFeatures.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/3/15.
//
//

#ifndef __Robust_tracking_by_detection__HoGandRawFeatures__
#define __Robust_tracking_by_detection__HoGandRawFeatures__

#include <stdio.h>
#include "Feature.h"
#include "RawFeatures.h"

class HoGandRawFeatures:public  Feature{
    
    cv::Size size;
    RawFeatures* rawFeatures;
    
    cv::HOGDescriptor* d;
public:
    HoGandRawFeatures(cv::Size hogSize_,int rawSize);
    
    cv::Mat prepareImage(cv::Mat* imageIn);
    
    arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
    
    int calculateFeatureDimension();
    
    std::string getInfo();

    ~HoGandRawFeatures(){
        delete rawFeatures;
        delete d;
    };
    
};

#endif /* defined(__Robust_tracking_by_detection__HoGandRawFeatures__) */
