//
//  ApproximateKernel.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//

#include "ApproximateKernel.h"
#include <algorithm>



void ApproximateKernel::preprocess(std::vector<supportData *> &S,int B){
    using namespace arma;
    
    colvec beta(B,arma::fill::zeros);
    
    // get size
    uword n=S[0]->x->n_cols;
    
    mat X(B,n,arma::fill::zeros);
    
    // iterate over all non-zero betas and add them into beta column vector
    // Also, populate non-sorted matrix X
    int idx=0;
    for (int i=0; i<S.size(); i++) {
        if (idx>=B) {
            break;
        }
        
        mat* b=S[i]->beta;
        for (int j=0; j<b->n_cols; j++) {
            if (b->at(0, j)!=0) {
                beta(idx)=b->at(0,j);
                X.row(idx)=S[i]->x->row(j);
                idx++;
            }
        }
    }
    //FIXME: approximate kernel not working. Needs rework.
    //TODO: Needs testing. Behavior of the approximate kernel is not satisfactory. There are bugs
//    this->threshold=idx;
//    if (idx<=B/2) {
//        return;
//    }
    
    preprocessMatrices(X, beta);
}

void ApproximateKernel::preprocessMatrices(arma::mat &X, arma::colvec &beta){

    using namespace arma;
    // for every i in n there should be a function approximated by a spline
    
    int m=X.n_rows;
    int n=X.n_cols;
    
    // find maximums and minimums
    
    rowvec mins=min(X,0);
    rowvec maxs=max(X,0);
    
    for (int i=0; i<n; i++) {
        
        double minVal=mins(i);
        double maxVal=maxs(i);

        rowvec y(this->nPts,fill::zeros);
        
        rowvec sample=linspace<rowvec>(minVal, maxVal,this->nPts);

        for (int j=0; j<this->nPts; j++) {
            for (int k=0; k<m; k++) {

                y(j)+=beta(k)*std::min(X(k,i), sample(j));
            }
        }
        
        Spline s;
        
        s.fitSpline(sample, y);
        this->splines.push_back(s);
        
    }
    
    
}


arma::rowvec ApproximateKernel::predictAll(arma::mat &newX, std::vector<supportData *> &S,int B){
    
  
    this->preprocess(S,B);
    
//    if (this->threshold<=B/2) {
//        return this->kernel->predictAll(newX, S, B);
//    }
    
    int nRows=newX.n_rows;
    arma::rowvec c(nRows,arma::fill::zeros);
    for (int i=0; i<nRows; i++) {
        arma::rowvec x=newX.row(i);
        
      
        
        c(i)=predictOne(x);
    }
    
    return c;
    
}

double ApproximateKernel::predictOne(arma::rowvec &x){
    
    double result=0;
    for (int i=0; i<x.size(); i++) {
        //std::cout<<i<<std::endl;
        result+=this->splines[i].evaluate(x(i));
    }
    
    return result;
}
