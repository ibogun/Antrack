//
//  MultiKernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/8/15.
//
//

#ifndef __Robust_tracking_by_detection__MultiKernel__
#define __Robust_tracking_by_detection__MultiKernel__

#include <stdio.h>
#include "Kernel.h"
#include <vector>
#include "../Features/Feature.h"

class MultiKernel:public Kernel{
    
    std::vector<int> featureDimensions;
    std::vector<Kernel*> kernels;
    std::vector<Feature*> features;
    
public:
    
    MultiKernel(std::vector<Kernel*>& k,std::vector<Feature*>& f){
        this->kernels=k;
        this->features=f;
    };
    
    void preprocess(std::vector<supportData*> & S, int B){};
    
    double calculate(arma::mat& x, int r1, arma::mat& x2, int r2);
    
    std::string getInfo();
    
    arma::rowvec predictAll(arma::mat& newX, std::vector<supportData*>& S, int B);
    
};
#endif /* defined(__Robust_tracking_by_detection__MultiKernel__) */
