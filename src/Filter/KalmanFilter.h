//
//  KalmanFilter.h
//  Structured_BING
//
//  Created by Ivan Bogun on 8/28/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Structured_BING__KalmanFilter__
#define __Structured_BING__KalmanFilter__

#include <iostream>
#include "armadillo"
#include <limits>
#include <algorithm>    // std::min
#include <opencv2/opencv.hpp>
class KalmanFilter_my {
    
    /*
    Notation and non-robust version of the filter is based on
     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
     */
    
    // initial parameters
    arma::mat F;                        // matrix which relates previous step to the current one
    arma::mat H;                        // matrix from measurements -> variables
    arma::mat Q;                        // process covariance matrix
    arma::mat R;                        // measurement covariance matrix
    int n;                              // dimension of the input vector (x_k)
    
    // time-varying parameters
    arma::mat P_kk;                     // a posteori estimate error covariance
    arma::mat K_kk;                     // Gain or blending factor
    
    int im_width;
    int im_height;
    
    // constant used for robustness of the filter, if infinity -> classical filter ( by default it is)
    double b;


    double b_given;
    
    friend std::ostream& operator<<(std::ostream &strm, const KalmanFilter_my &f);
    
public:
    
    arma::colvec x_kk;                  // current value of the variables
    
    KalmanFilter_my(){};
    
    cv::Rect getBoundingBox(int,int,arma::colvec);
    
    void setBothB(const double b_){this->b=b_; this->b_given=b_;};
    void setB(const double b_){this->b=b_;}

    double getGivenB(){
        return b_given;
    }
    
    KalmanFilter_my(arma::mat& F_,arma::mat& H_,arma::mat& Q_,arma::mat& R_,
                 arma::mat& P_0,arma::colvec& x_0,int im_w,int im_h,double b=std::numeric_limits<double>::infinity());
    
    arma::colvec predict(arma::colvec&);
    arma::colvec predictAndCorrect(arma::colvec&);
    
    
};

#endif /* defined(__Structured_BING__KalmanFilter__) */
