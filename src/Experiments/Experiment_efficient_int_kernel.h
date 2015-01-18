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
#include "../Kernels/ApproximateKernel.h"
#include "../Kernels/IntersectionKernel_additive.h"

class ExperimentEfficientIntersectionKernel {
    
    IntersectionKernel kernel_simple;
    IntersectionKernel_fast kernel_fast;
    
    ApproximateKernel* kernel_approx;
    
    arma::mat X;
    arma::colvec beta;
    arma::mat x_test;
    
public:
    
    
    ExperimentEfficientIntersectionKernel(int n,int m, int nTestCases, int approxPts);
    
    float calculateTimeRegularKernel();
    std::pair<float, float> calculateTimeFastKernel();
    std::pair<float, float>calculateTimeApproxKernel();
    
    static void performExperiment(std::string outputDir);
};

#endif /* defined(__Robust_tracking_by_detection__Experiment_efficient_int_kernel__) */
