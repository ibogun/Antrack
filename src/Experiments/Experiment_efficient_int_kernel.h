//
//  Experiment_efficient_int_kernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#ifndef __Robust_tracking_by_detection__Experiment_efficient_int_kernel__
#define __Robust_tracking_by_detection__Experiment_efficient_int_kernel__

#include <stdio.h>
#include "../Kernels/IntersectionKernel.h"
#include "../Kernels/IntersectionKernel_fast.h"

class ExperimentEfficientIntersectionKernel {
    
    IntersectionKernel kernel_simple;
    IntersectionKernel_fast kernel_fast;
    
    arma::mat X;
    arma::colvec beta;
    arma::mat x_test;
    
public:
    
    
    ExperimentEfficientIntersectionKernel(int n,int m, int nTestCases);
    
    float calculateTimeRegularKernel();
    float calculateTimeLinearKernel();
    std::pair<float, float> calculateTimeFastKernel();
    
    static void performExperiment(std::string outputDir);
};

#endif /* defined(__Robust_tracking_by_detection__Experiment_efficient_int_kernel__) */
