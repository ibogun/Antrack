//
//  Histogram.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Histogram__
#define __Robust_Struck__Histogram__

#include <stdio.h>
#include "Feature.h"

class HistogramFeatures:public  Feature{
    int L;
    
    int NUMBER_OF_BINS_PER_HISTOGRAM;
public:
    HistogramFeatures(int L_,int nBins){ this->L=L_;NUMBER_OF_BINS_PER_HISTOGRAM=nBins;};
    
    cv::Mat prepareImage(cv::Mat* imageIn);

    arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
    
    int calculateFeatureDimension(){
        int S=0;
        for (int i=1; i<=this->L; i++) {
            S=S+i*i;
        }
        return S*NUMBER_OF_BINS_PER_HISTOGRAM;
    };
    
    std::string getInfo(){
        std::string r="Histogram features with \n";
        r+="L                 : "+std::to_string(L)+"\n";
        r+="BINS/Histogram    : "+std::to_string(NUMBER_OF_BINS_PER_HISTOGRAM)+"\n";
        
        return r;
    };

    ~HistogramFeatures(){}
    
};

#endif /* defined(__Robust_Struck__Histogram__) */
