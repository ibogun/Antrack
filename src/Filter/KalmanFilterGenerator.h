//
//  KalmanFilterGenerator.h
//  Structured_BING
//
//  Created by Ivan Bogun on 8/28/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Structured_BING__KalmanFilterGenerator__
#define __Structured_BING__KalmanFilterGenerator__

#include <iostream>
#include "KalmanFilter.h"

class KalmanFilterGenerator {
    
    
public:
    
    static KalmanFilter_my generateConstantVelocityFilter(arma::colvec x_0,int im_w,
                                                          int im_h,double q, double r,
                                                          double p, double b=std::numeric_limits<double>::infinity());
    
    static KalmanFilter_my generateConstantAccelerationFilter(arma::colvec x_0,int im_w,int im_h,
                                                              double q, double r, double p,
                                                              double b=std::numeric_limits<double>::infinity());
    
    static KalmanFilter_my generateConstantVelocityWithScaleFilter(arma::colvec x_0,int im_w,int im_h,
                                                                   double q, double r, double p,
                                                                   double b=std::numeric_limits<double>::infinity());

    static KalmanFilter_my generateFilterCenterTranslation(arma::colvec x_0,int im_w,int im_h,
                                                                   double q, double r, double p,
                                                                   double b=std::numeric_limits<double>::infinity());

    static KalmanFilter_my generateFilterScaleChange(arma::colvec x_0,int im_w,int im_h,
                                                           double q, double r, double p,
                                                           double b=std::numeric_limits<double>::infinity());
    
};

#endif /* defined(__Structured_BING__KalmanFilterGenerator__) */
