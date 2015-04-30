//
//  ApproximateKernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//

#ifndef __Robust_tracking_by_detection__ApproximateKernel__
#define __Robust_tracking_by_detection__ApproximateKernel__

#include <stdio.h>
#include "Kernel.h"
#include "AdditiveKernel.h"
#include "IntersectionKernel_fast.h"
#include "IntersectionKernel.h"
#include "armadillo"
#include "Spline.h"
class ApproximateKernel: public Kernel{

    Kernel* kernel;
    int nPts;
    
    std::vector<Spline> splines;
    int threshold;
    
public:
    ApproximateKernel(int nPts_,Kernel* k_){ nPts=nPts_;kernel=k_;};
    
    void preprocess(std::vector<supportData*>& S,int B);
    void preprocessMatrices(arma::mat& X, arma::colvec& beta);

    double calculate(arma::mat& x,int r1,arma::mat& x2,int r2){
        return kernel->calculate(x, r1, x2, r2);
    };
    
    std::string getInfo(){
        std::string r="Approximate Intersection kernel with "+std::to_string(nPts)+" points";
        return r;
    };
    
    
    
    
    double predictOne(arma::rowvec& x);
    arma::rowvec predictAll(arma::mat& newX,std::vector<supportData*>& S,int B);


    ~ApproximateKernel(){
        delete kernel;
    }
    
};

#endif /* defined(__Robust_tracking_by_detection__ApproximateKernel__) */
