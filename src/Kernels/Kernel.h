//
//  Kernel.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Kernel__
#define __Robust_Struck__Kernel__
#include <opencv2/opencv.hpp>
#include "armadillo"

class Kernel {
    
    
public:

    ~Kernel(){}
    virtual float calculateKernelValue(float* x1, float* x2, int size)=0;
    virtual double calculate(arma::mat& x,int r1,arma::mat& x2,int r2)=0;
    
    
};

#endif /* defined(__Robust_Struck__Kernel__) */
