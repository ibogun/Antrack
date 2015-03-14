//
//  MultiKernel.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/8/15.
//
//

#include "MultiKernel.h"




double MultiKernel::calculate(arma::mat &x, int r1, arma::mat &x2, int r2){
    

    int dim1=x.n_rows;
    int dim2=x2.n_rows;
    int l = 0,h=0;
    
    
    double result=0;
    
    
    for (int i=0; i<features.size(); i++) {
        
        h+=features[i]->calculateFeatureDimension();
        
        

        arma::mat x_1_part=x.submat(0, l, dim1-1, h-1);
        arma::mat x_2_part=x2.submat(0, l, dim2-1, h-1);
        
        
        result+=this->kernels[i]->calculate(x_1_part, r1, x_2_part, r2);
        l=h;
    }
    
    return result;
    
}

std::string MultiKernel::getInfo(){
    std::string r="";
    
    for (int i=0; i<this->kernels.size(); i++) {
        r+=this->kernels[i]->getInfo();
    }
    return r;
}


arma::rowvec MultiKernel::predictAll(arma::mat &newX,std::vector<supportData*>& S,int B){
    using namespace arma;
    
    // preprocess first
    preprocess(S,B);
    
    int n=newX.n_rows;
    
    mat y_hat(1,1,fill::zeros);
    mat y(1,1,fill::zeros);
    
    rowvec scores(n,fill::ones);
    
    int dim1=newX.n_rows;
    int l = 0,h=0;
    std::vector<arma::mat> new_x;
    
    for (int i=0; i<this->features.size(); i++) {
        h+=features[i]->calculateFeatureDimension();
        arma::mat x_1_part=newX.submat(0, l, dim1-1, h-1);
        new_x.push_back(x_1_part);
        l=h;
    }
    
    for (int k = 0; k < newX.n_rows; ++k) {
        
        y(0)=k;
        double current=0;
        
        
        for (int i = 0; i < S.size(); ++i) {
            int dim2=S[i]->x->n_rows;
            l=0;
            h=0;
            std::vector<arma::mat> old_x;
            
            
            for (int j=0; j<this->features.size(); j++) {
                h+=features[j]->calculateFeatureDimension();
                arma::mat x_2_part=S[i]->x->submat(0, l, dim2-1, h-1);
                old_x.push_back(x_2_part);
                l=h;
            }
            
            for (int yhat = 0; yhat < S[i]->x->n_rows; ++yhat) {
                
                y_hat(0)=yhat;
                
                if ((*S[i]->beta)[yhat]!=0){
                    
                    // the below has to be multiplied by the velocities kernel
//                    current+= (*S[i]->beta)[yhat]*
//                    calculate(newX, y(0), *S[i]->x, y_hat(0));
                    
                    for (int j=0; j<this->features.size(); j++) {
                        current+=(*S[i]->beta)[yhat]*this->kernels[j]->calculate(new_x[j], y(0), old_x[j], y_hat(0));
                    }
                    
                }
            }
            
        }
        
        scores[k]=current;
        
        
    }
    
    return scores;
}