//
//  Objectness.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/4/15.
//
//

#ifndef __Robust_tracking_by_detection__Objectness__
#define __Robust_tracking_by_detection__Objectness__

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "armadillo"
#include "SuperPixels.h"
class Straddling{
    
    
    int nSuperPixels;
    double inner_threshold;
    
    
    
public:
    

    Straddling(int n,double inner=0.9){this->nSuperPixels=n;
        inner_threshold=inner;};

    cv::Rect getInnerRect(cv::Rect& r);
    
    arma::mat getLabels(cv::Mat&);
    arma::rowvec findStraddling(arma::mat& labels,std::vector<cv::Rect>& rects, int translate_x, int translate_y);
    
    double findStraddlingMeasure(arma::mat& labels, cv::Rect& rectangle);
    
    arma::rowvec findStraddlng_fast(arma::mat& labels,std::vector<cv::Rect>& rects, int translate_x,int translate_y);
    
};


class EdgeDensity{
    
    double threshold_1;
    double threshold_2;
    
    double inner_threshold;
    
public:
    EdgeDensity(double t1,double t2, double inner){this->threshold_1=t1; this->threshold_2=t2;
        this->inner_threshold=inner;};
    
    cv::Mat getEdges(cv::Mat&);
    
    arma::rowvec findEdgeObjectness(cv::Mat& labels, std::vector<cv::Rect>& rects, int translate_x, int translate_y);
};

#endif /* defined(__Robust_tracking_by_detection__Objectness__) */
