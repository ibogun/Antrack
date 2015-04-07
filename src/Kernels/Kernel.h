//
//  Kernel.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Kernel__
#define __Robust_Struck__Kernel__
#include <opencv2/opencv.hpp>
#include "armadillo"
#include "../Tracker/supportData.h"

class Kernel {



    
public:

    virtual ~Kernel(){};
    
    
    
    virtual void preprocess(std::vector<supportData*>& S,int B)=0;
    virtual double calculate(arma::mat& x,int r1,arma::mat& x2,int r2)=0;
    virtual std::string getInfo()=0;
    
    virtual arma::rowvec predictAll(arma::mat& newX,std::vector<supportData*>& S, int B);
    
    
};

#endif /* defined(__Robust_Struck__Kernel__) */
