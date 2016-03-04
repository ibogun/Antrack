//
//  LinearKernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/27/15.
//
//

#ifndef __Robust_tracking_by_detection__LinearKernel__
#define __Robust_tracking_by_detection__LinearKernel__

#include <stdio.h>
#include "Kernel.h"

class LinearKernel:public Kernel {
    
    
public:
    
    void preprocess(std::vector<supportData*>& S,int B){};
    
     double calculate(arma::mat& x1, int r1, arma::mat& x2, int r2){
        
        //arma::mat row1=x1.row(r1);
        //arma::mat row2=x2.row(r2);
        
        return arma::dot(x1.row(r1),x2.row(r2));
    };
    
    
    std::string getInfo(){
        return "Linear kernel";
    }

    ~LinearKernel(){
    }
    
};

#endif /* defined(__Robust_tracking_by_detection__LinearKernel__) */
