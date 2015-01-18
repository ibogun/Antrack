//
//  IntersectionKernel_additive.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/24/14.
//
//

#ifndef __Robust_tracking_by_detection__IntersectionKernel_additive__
#define __Robust_tracking_by_detection__IntersectionKernel_additive__

#include <stdio.h>
#include "AdditiveKernel.h"
#include <algorithm>
class IntersectionKernel_additive:public AdditiveKernel{
    
    
public:
    
    double calculate(double x1,double x2){return std::min(x1,x2);};
};

#endif /* defined(__Robust_tracking_by_detection__IntersectionKernel_additive__) */
