//
//  Haar.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Haar__
#define __Robust_Struck__Haar__

#include <stdio.h>
#include "Feature.h"
class Haar:public Feature {
    int scale;
    
    const int dimPerScale=112;
public:
    Haar(int scale_){this->scale=scale_;};
    
    cv::Mat prepareImage(cv::Mat* imageIn);
    arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
    int calculateFeatureDimension(){return scale*dimPerScale;};
    
    arma::vec calculateHaarFeature(cv::Mat& integral_box,int gridHeight, int gridLength,const int normalize);
    std::vector<int> linspace(double,double,int);
    
    static int round_my(double);
    
    double getSumGivenCorners(cv::Mat& integral,const int& xmin,const int& xmax,const int& ymin,const int& ymax);
    
    std::string getInfo();
};

#endif /* defined(__Robust_Struck__Haar__) */
