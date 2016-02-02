//
//  IntersectionKernelTest.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//

#ifndef __Robust_tracking_by_detection__IntersectionKernelTest__
#define __Robust_tracking_by_detection__IntersectionKernelTest__
#include "../Kernels/IntersectionKernel.h"
#include "../Kernels/IntersectionKernel_fast.h"

#include "../Features/Histogram.h"
#include "../Tracker/Struck.h"
#include "../Tracker/LocationSampler.h"

#include "../Datasets/DatasetWu2013.h"
#include "../Datasets/DatasetVOT2014.h"
#include "../Datasets/DatasetALOV300.h"


#include <stdio.h>
class IntersectionKernel_tracking_test : public ::testing::Test {
    
public:

    
    IntersectionKernel_tracking_test(){}
    
    
    virtual ~IntersectionKernel_tracking_test(){
        
    };
    
    virtual void setUp(){
        
 
       
    }
    
};

#endif /* defined(__Robust_tracking_by_detection__IntersectionKernelTest__) */
