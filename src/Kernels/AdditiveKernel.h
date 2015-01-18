//
//  AdditiveKernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/24/14.
//
//

#ifndef __Robust_tracking_by_detection__AdditiveKernel__
#define __Robust_tracking_by_detection__AdditiveKernel__

#include <stdio.h>

class AdditiveKernel {
    
    
public:
    virtual double calculate(double, double)=0;
};

#endif /* defined(__Robust_tracking_by_detection__AdditiveKernel__) */
