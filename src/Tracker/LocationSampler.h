//
//  LocationSampler.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__LocationSampler__
#define __Robust_Struck__LocationSampler__

#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../Features/Haar.h"

class LocationSampler {
     int radius;

    
    int objectHeight;
    int objectWidth;
    
    int n=0;
    int m=0;
    
    cv::Rect fromCenterToBoundingBox(const double&,const double&,const double&, const double&);
    
public:
    
    int nRadial;
    int nAngular;
    
    LocationSampler(int r,int nRad, int nAng)
    : radius(r),nRadial(nRad), nAngular(nAng){}
    
    void sampleOnAGrid(cv::Rect& currentRect,std::vector<cv::Rect>& rects, int R,int distance=1);
    
    void sampleEquiDistant(cv::Rect& currentRect,std::vector<cv::Rect>& rects);
    
     std::vector<double> linspace(double a,double b, double n);
    
    void setDimensions(int imN,int imM, int objH, int objW){
        n=imN;
        m=imM;
        objectWidth=objW;
        objectHeight=objH;
    }
    
    
    
    
};

#endif /* defined(__Robust_Struck__LocationSampler__) */
