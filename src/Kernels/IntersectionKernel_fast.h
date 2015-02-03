//
//  IntersectionKernel_fast.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#ifndef __Robust_tracking_by_detection__IntersectionKernel_fast__
#define __Robust_tracking_by_detection__IntersectionKernel_fast__

#include <stdio.h>


#include "Kernel.h"
#include "armadillo"

class IntersectionKernel_fast:public Kernel {
    
    arma::mat A;
    arma::mat B;
    arma::mat h;
    arma::mat x_s;
    
public:
    
    IntersectionKernel_fast(){};
    
    void preprocess(std::vector<supportData*>& S,int B);
    void preprocessMatrices(arma::mat& X, arma::colvec& beta);
    
    float calculateKernelValue(float* x1, float* x2, int size);
    double calculate(arma::mat& x,int r1,arma::mat& x2,int r2);
    
    int binarySearch(const arma::colvec& x,double z);
    arma::rowvec predictAll(arma::mat& newX,std::vector<supportData*>& S,int B);
    
    double predictOne(arma::rowvec& x);
    
    
    std::string getInfo(){
        std::string r="Fast exact Intersection kernel";
        return r;
    };
};
#endif /* defined(__Robust_tracking_by_detection__IntersectionKernel_fast__) */
