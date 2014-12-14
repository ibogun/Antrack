//
//  KalmanFilterGenerator.cpp
//  Structured_BING
//
//  Created by Ivan Bogun on 8/28/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "KalmanFilterGenerator.h"



KalmanFilter_my KalmanFilterGenerator::generateConstantVelocityFilter(arma::colvec x_0,int im_w,int im_h,double q, double r, double p, double b){
    
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


KalmanFilter_my KalmanFilterGenerator::generateConstantAccelerationFilter(arma::colvec x_0,int im_w,int im_h,double q, double r, double p, double b){
    
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