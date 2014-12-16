//
//  IntersectionKernel.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "IntersectionKernel.h"
#include <algorithm>    // std::min

float IntersectionKernel::calculateKernelValue(float *x1, float *x2, int size){
 
    float result=0;
    
    
    // wasn't tested
    for (int i=0; i<size; i++) {
        result+=std::min(*(x1+i),*(x2+i));
    }
    
    return result;
}

double IntersectionKernel::calculate(arma::mat &x, int r1, arma::mat &x2, int r2){
   
    
    arma::mat combinedX=arma::join_vert(x.row(r1), x2.row(r2));
    
    arma::mat t=arma::min(combinedX);
    double r=arma::sum(arma::sum(t));
//     double result=0;
//    for (int i=0; i<x.n_cols; i++) {
//        result+=std::min(x(r1,i),x2(r2,i));
//    }
//    result=result/x.n_cols;
    return r;
}