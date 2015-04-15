//
//  KalmanFilterGenerator.cpp
//  Structured_BING
//
//  Created by Ivan Bogun on 8/28/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "KalmanFilterGenerator.h"

KalmanFilter_my KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(arma::colvec x_0,
                                                                               int im_w,int im_h,
                                                                               double q, double r,
                                                                               double p, double b){
    
    using namespace arma;
    
    mat F = eye<mat>(10,10);
    
    F(0,4)=1;
    F(1,5)=1;
    F(2,6)=1;
    F(3,7)=1;
    
    F(0,8)=1;
    F(2,8)=1;
    F(1,9)=1;
    F(3,9)=1;
    
    mat H=eye<mat>(4,10);
    
    mat Q=eye<mat>(10,10);
    
    for (int i=0; i<=3; i++) {
        Q(i,i)=1.0/4;
    }
    
    Q(0,4)=1;
    Q(0,6)=1;
    Q(1,5)=1;
    Q(1,7)=1;
    Q(2,6)=1;
    Q(2,4)=1;
    Q(3,7)=1;
    Q(3,5)=1;
    
    Q(0,8)=1;
    Q(2,8)=1;
    Q(1,9)=1;
    Q(3,9)=1;
    
    for (int i=0; i<3; i++) {
        Q(4+i*2,0)=1.0/2;
        Q(4+i*2,2)=1.0/2;
        Q(5+i*2,1)=1.0/2;
        Q(5+i*2,3)=1.0/2;
    }
 
    
 
    
    Q=Q*q;
    
    
    mat R=arma::eye(4, 4);
    R=R*r;
    
    mat P=arma::eye(10, 10);
    P=P*p;
    
    KalmanFilter_my filter(F, H, Q, R, P,x_0,im_w,im_h, b);
    
    return filter;
}


KalmanFilter_my KalmanFilterGenerator::generateConstantVelocityFilter(arma::colvec x_0,int im_w,
                                                                      int im_h,double q, double r,
                                                                      double p, double b){
    
    using namespace arma;
    
    mat F(6,6,fill::zeros);
    F(0,0)=1;
    F(1,1)=1;
    F(2,2)=1;
    F(3,3)=1;
    F(4,4)=1;
    F(5,5)=1;
    
    F(0,4)=1;
    F(1,5)=1;
    F(2,4)=1;
    F(3,5)=1;
    
    mat H(4,6,fill::zeros);
    H(0,0)=1;
    H(1,1)=1;
    H(2,2)=1;
    H(3,3)=1;
    
    
    mat Q(6,6,fill::zeros);
    
    Q(0,0)=1.0/4;
    Q(1,1)=1.0/4;
    Q(2,2)=1.0/4;
    Q(3,3)=1.0/4;
    Q(4,4)=1;
    Q(5,5)=1;
    
    Q(0,4)=1.0/2;
    Q(1,5)=1.0/2;
    Q(2,4)=1.0/2;
    Q(3,5)=1.0/2;
    
    Q(4,0)=1.0/2;
    Q(4,2)=1.0/2;
    Q(5,1)=1.0/2;
    Q(5,3)=1.0/2;
    
    Q=Q*q;
    
    
    mat R=arma::eye(4, 4);
    R=R*r;
    
    mat P=arma::eye(6, 6);
    P=P*p;
    
    KalmanFilter_my filter(F, H, Q, R, P,x_0,im_w,im_h, b);

    return filter;
}


KalmanFilter_my KalmanFilterGenerator::generateConstantAccelerationFilter(arma::colvec x_0,
                                                                          int im_w,int im_h,
                                                                          double q, double r,
                                                                          double p, double b){
    
    using namespace arma;
    
    int n=8;
    
    mat F(n,n,fill::zeros);

    F=eye(n, n);
    
    F(0,4)=1;
    F(1,5)=1;
    F(2,4)=1;
    F(3,5)=1;
    
    F(0,6)=0.5;
    F(1,7)=0.5;
    F(2,6)=0.5;
    F(3,7)=0.5;
    
    mat H(4,n,fill::zeros);
    H(0,0)=1;
    H(1,1)=1;
    H(2,2)=1;
    H(3,3)=1;
    
    
    mat Q(n,n,fill::zeros);
    
    Q(0,0)=1.0/4;
    Q(1,1)=1.0/4;
    Q(2,2)=1.0/4;
    Q(3,3)=1.0/4;
    Q(4,4)=1;
    Q(5,5)=1;
    
    Q(0,4)=1.0/2;
    Q(1,5)=1.0/2;
    Q(2,4)=1.0/2;
    Q(3,5)=1.0/2;
    
    Q(0,6)=1;
    Q(1,7)=1;
    Q(2,6)=1;
    Q(3,7)=1;
    
    
    Q(4,0)=1.0/2;
    Q(4,2)=1.0/2;
    Q(5,1)=1.0/2;
    Q(5,3)=1.0/2;
    
    Q(4,6)=1;
    Q(5,7)=1;
    
    
    
    Q(6,6)=1;
    
    Q(7,7)=1;
    
    Q=Q*q;
    
    
    mat R=arma::eye(4, 4);
    R=R*r;
    
    mat P=arma::eye(n, n);
    P=P*p;
    
    KalmanFilter_my filter(F, H, Q, R, P, x_0,im_w,im_h, b);
    
    return filter;
}


KalmanFilter_my KalmanFilterGenerator::generateFilterCenterTranslation(arma::colvec x_0, int im_w, int im_h, double q,
                                                                       double r, double p, double b) {


    using namespace arma;

    mat F(4,4,fill::zeros);
    F(0,0)=1;
    F(1,1)=1;
    F(2,2)=1;
    F(3,3)=1;
    F(0,2)=1;
    F(1,3)=1;

    mat H(2,4,fill::zeros);
    H(0,0)=1;
    H(1,1)=1;


    mat Q(4,4,fill::zeros);

    Q(0,0)=1.0/4;
    Q(1,1)=1.0/4;
    Q(2,2)=1.0;
    Q(3,3)=1.0;

    Q(0,2)=1.0/2;
    Q(1,3)=1.0/2;

    Q(2,0)=1.0/2;
    Q(3,1)=1.0/2;

    Q=Q*q;


    mat R=arma::eye(2, 2);
    R=R*r;

    mat P=arma::eye(4, 4);
    P=P*p;

    KalmanFilter_my filter(F, H, Q, R, P,x_0,im_w,im_h, b);

    return filter;

}


KalmanFilter_my KalmanFilterGenerator::generateFilterScaleChange(arma::colvec x_0, int im_w, int im_h, double q,
                                                                 double r, double p, double b) {


    using namespace arma;

    mat F(4,4,fill::zeros);
    F(0,0)=1;
    F(1,1)=1;
    F(2,2)=1;
    F(3,3)=1;
    F(0,2)=1;
    F(1,3)=1;

    mat H(2,4,fill::zeros);
    H(0,0)=1;
    H(1,1)=1;


    mat Q(4,4,fill::zeros);

    Q(0,0)=1.0/4;
    Q(1,1)=1.0/4;
    Q(2,2)=1.0;
    Q(3,3)=1.0;

    Q(0,2)=1.0/2;
    Q(1,3)=1.0/2;

    Q(2,0)=1.0/2;
    Q(3,1)=1.0/2;

    Q=Q*q;


    mat R=arma::eye(2, 2);
    R=R*r;

    mat P=arma::eye(4, 4);
    P=P*p;

    KalmanFilter_my filter(F, H, Q, R, P,x_0,im_w,im_h, b);

    return filter;
}