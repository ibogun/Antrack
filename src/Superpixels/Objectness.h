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
    
public:
    Straddling(int n){this->nSuperPixels=n;};
    
    arma::mat getLabels(cv::Mat&);
    arma::rowvec findStraddling(arma::mat& labels,std::vector<cv::Rect>& rects, int translate_x, int translate_y);
    
    double findStraddlingMeasure(arma::mat& labels, cv::Rect& rectangle);
    
};

#endif /* defined(__Robust_tracking_by_detection__Objectness__) */
