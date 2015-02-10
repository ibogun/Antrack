//
//  Kernel.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Kernel.h"


arma::rowvec Kernel::predictAll(arma::mat &newX,std::vector<supportData*>& S,int B){
    using namespace arma;
    
    // preprocess first
    preprocess(S,B);
    
    int n=newX.n_rows;
    
    mat y_hat(1,1,fill::zeros);
    mat y(1,1,fill::zeros);
    
    rowvec scores(n,fill::ones);
    
    for (int k = 0; k < newX.n_rows; ++k) {
        
        y(0)=k;
        double current=0;
        
        
        for (int i = 0; i < S.size(); ++i) {
            for (int yhat = 0; yhat < S[i]->x->n_rows; ++yhat) {
                
                y_hat(0)=yhat;
                
                if ((*S[i]->beta)[yhat]!=0){
                    
                    // the below has to be multiplied by the velocities kernel
                    current+= (*S[i]->beta)[yhat]*
                    calculate(newX, y(0), *S[i]->x, y_hat(0));
                    
                }
            }
            
        }
        
        scores[k]=current;
        
        
    }
    
    return scores;
}