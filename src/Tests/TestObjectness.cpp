//
//  TestObjectness.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/9/15.
//
//

#include "TestObjectness.h"




TEST_F(TestObjectness,findStraddeling){
    
    int superpixels=25;
    double inner=0.9;
    Straddling straddeling(superpixels,inner);
    
    arma::mat labels=straddeling.getLabels(image);
    
    arma::rowvec r1=straddeling.findStraddling(labels, rects, 0, 0);
    
    arma::rowvec r2=straddeling.findStraddlng_fast(labels, rects, 0, 0);
    
    
    EXPECT_NEAR(arma::norm(r1-r1), 0, 1e-6);
    
}