//
//  RBFKernel.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__RBFKernel__
#define __Robust_Struck__RBFKernel__

#include <stdio.h>
#include "Kernel.h"



class RBFKernel:public Kernel {
    
    float gamma;
    
public:
    RBFKernel(float gamma_){ this->gamma=gamma_;};
    float calculateKernelValue(float* x1, float* x2, int size);
    double calculate(arma::mat& x1,int row1,arma::mat& x2, int row2);
};

#endif /* defined(__Robust_Struck__RBFKernel__) */
