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

    int n=0;
    int m=0;

    cv::Rect fromCenterToBoundingBox(const double&,const double&,const double&,
                                     const double&);

    friend std::ostream& operator<<(std::ostream&,const  LocationSampler&);

public:

    int objectHeight;
    int objectWidth;

    int nRadial;
    int nAngular;
    double downsample = 1.05;
    int min_scales = -4;
    int max_scales = 8;
    int shrink_one_side_scale = 2;

    int min_size_half = 10; // minumum size of the side

LocationSampler(int r,int nRad, int nAng)
    : radius(r),nRadial(nRad), nAngular(nAng){}

    void sampleOnAGrid(cv::Rect& currentRect,std::vector<cv::Rect>& rects,
                       int R,int distance=1);

    int getRadius(){
        return this->radius;
    }
    template <typename T>
 static std::vector<size_t> sort_indexes(const std::vector<T> &v) ;

    void rearrangeByDimensionSimilarity(const cv::Rect& rect,
    std::vector<int>& radiuses, std::vector<int>& widths,
    std::vector<int>& heights);

    void sampleEquiDistant(cv::Rect& currentRect,std::vector<cv::Rect>& rects);

    void sampleEquiDistantMultiScale(cv::Rect& currentRect,
                                     std::vector<cv::Rect>& rects);

    std::vector<double> linspace(double a,double b, double n);

    void setDimensions(int imN,int imM, int objH, int objW){
        n=imN;
        m=imM;
        objectWidth=objW;
        objectHeight=objH;
    }

    static std::vector<arma::mat> generateBoxesTensor(
        const int R,
        const int scale_R,
        const int min_size_half,
        const int min_scales,
        const int max_scales,
        const double downsample,
        const double shrink_one_side_scale,
        const int rows,
        const int cols,
        const cv::Rect& rect,
        std::vector<int>* radiuses,
        std::vector<int>* widths,
        std::vector<int>* heights);

    std::vector<arma::mat> generateBoxesTensor(const cv::Rect& rect,
                                               std::vector<int>* radiuses,
                                               std::vector<int>* widths,
                                               std::vector<int>* heights);
};

#endif /* defined(__Robust_Struck__LocationSampler__) */
