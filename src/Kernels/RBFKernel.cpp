//
//  RBFKernel.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "RBFKernel.h"
#include <cmath>    // std::min

float RBFKernel::calculateKernelValue(float *x1, float *x2, int size){
    float result=0;
    
    
    // wasn't tested
    for (int i=0; i<size; i++) {
        result+=powf(*(x1+i)- *(x2+i),2);
    }
    result=expf(-this->gamma*result);
    
    return  result;
}


double RBFKernel::calculate(arma::mat& x1, int row1, arma::mat& x2, int row2){
    
    //float result=0;
    



    
    
    double result=arma::sum(arma::pow(x1.row(row1)-x2.row(row2),2));
    
    // wasn't tested
//    double res=0;
//    int size=x1.n_cols;
    //    for (int i=0; i<size; i++) {
    //        res+=pow(x1(row1,i)-x2(row2,i),2);
    //    }
//    if (abs(res-result)>0.0001) {
//        std::cout<<"Not equal: "<<result<<" "<<res<<std::endl;
//    }
    
    result=exp(-this->gamma*result);
    
    return  result;
}