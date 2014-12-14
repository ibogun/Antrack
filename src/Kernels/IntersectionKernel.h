//
//  IntersectionKernel.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__IntersectionKernel__
#define __Robust_Struck__IntersectionKernel__

#include <stdio.h>
#include "Kernel.h"

class IntersectionKernel:public Kernel {
    
    
public:
    
    IntersectionKernel(){};
    float calculateKernelValue(float* x1, float* x2, int size);
    double calculate(arma::mat& x,int r1,arma::mat& x2,int r2);
};


#endif /* defined(__Robust_Struck__IntersectionKernel__) */