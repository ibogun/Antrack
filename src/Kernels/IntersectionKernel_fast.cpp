//
//  IntersectionKernel_fast.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#include "IntersectionKernel_fast.h"


void IntersectionKernel_fast::preprocess(std::vector<supportData *> &S,int B){
    
    using namespace arma;
    
    // BALANCE should be here
    colvec beta(B,arma::fill::zeros);
    
    // get size
    uword n=S[0]->x->n_cols;
    
    mat X(B,n,arma::fill::zeros);
    
    // iterate over all non-zero betas and add them into beta column vector
    // Also, populate non-sorted matrix X
    int idx=0;
    for (int i=0; i<S.size(); i++) {
        mat* b=S[i]->beta;
        for (int j=0; j<b->n_cols; j++) {
            if (b->at(0, j)!=0) {
                beta(idx)=b->at(0,j);
                X.row(idx)=S[i]->x->row(j);
                idx++;
            }
        }
    }
    
    preprocessMatrices(X, beta);

    
}

void IntersectionKernel_fast::preprocessMatrices(arma::mat &X, arma::colvec &beta){
    using namespace arma;
    
    int m=X.n_rows;
    int n=X.n_cols;
    
    // allocate objects
    
    mat A(m,n,arma::fill::zeros);
    
    mat B(m,n,arma::fill::zeros);
    mat h(m,n,arma::fill::zeros);
    
    
    mat x_s(m,n,arma::fill::zeros);
    
    // sort X and get indices
    // unfortunately there is no function which can sort AND return indices, thus
    // sorting twice
    
    x_s=sort(X,"ascend",0);
    
    Mat<uword> I(m,n);
    for (int i=0; i<n; i++) {
        I.col(i)=sort_index(X.col(i),"ascend");

    }
    
    mat Y(m,n);
    
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            Y(i,j)=beta(I(i,j));
        }
    }
    
    /*
     Precompute A,B,h
     for i=1:n
     A(1,i)=Y(1,i)*x_s(1,i);
     
     for r=2:m
     A(r,i)=A(r-1,i)+Y(r,i)*x_s(r,i);
     end
     
     for r=m-1:-1:1
     B(r,i)=B(r+1,i)+Y(r+1,i);
     end
     end
     
     
     h=zeros(m,n);
     
     for i=1:n
     for r=1:m
     
     h(r,i)=A(r,i)+x_s(r,i)*B(r,i);
     end
     end
     
     */
    
    for (int i=0; i<n; i++) {
        A(0,i)=Y(0,i)*x_s(0,i);
        for (int r=1; r<m; r++) {
            A(r,i)=A(r-1,i)+Y(r,i)*x_s(r,i);
        }
        
        for (int r=m-2; r>=0; r--) {
            B(r,i)=B(r+1,i)+Y(r+1,i);
        }
    }
    
    for (int i=0; i<n; i++) {
        for (int r=0; r<m; r++) {
            h(r,i)=A(r,i)+x_s(r,i)*B(r,i);
        }
    }
    
    
    this->h=h;
    this->x_s=x_s;
}

arma::rowvec IntersectionKernel_fast::predictAll(arma::mat &newX, std::vector<supportData *> &S,int B){
    
    this->preprocess(S,B);
    
    int nRows=newX.n_rows;
    arma::rowvec c(nRows,arma::fill::zeros);
    for (int i=0; i<nRows; i++) {
        arma::rowvec x=newX.row(i);
        
        c(i)=predictOne(x);
    }
    
    return c;
}


double IntersectionKernel_fast::predictOne(arma::rowvec &x){
    
    double k=0;
    double v=0;
    int K=this->x_s.n_rows-1;
    
    double low, high;
    for (int i=0; i<x.size(); i++) {
        
        v=0;
        int r=binarySearch(this->x_s.col(i), x(i));
        
        if (x(i)<x_s(0,i)) {
            v=this->h(r,i);
        }else if (x(i)>=x_s(K,i)){
            v=h(r,i);
        }else{
            low=h(r,i);
            high=h(r+1,i);
            v=((high-low)/(x_s(r+1,i)-x_s(r,i)))*(x(i)-x_s(r,i))+low;
        }
        
        k+=v;
        /*
         
         v=0;
         [~,r]=binarySearch(x_s(:,i),x1(i));
         
         if (r~=1)
         assert(x_s(r,i)<=x1(i));
         end
         if (x1(i)<x_s(1,i))
         v=0;
         elseif(x1(i)>=x_s(m,i))
         v=h(r,i);
         
         else
         %         low=A(r,i)+x_s(r,i)*B(r,i);
         %         high=A(r,i)+x_s(r+1,i)*B(r,i);
         
         low=h(r,i);
         high=h(r+1,i);
         % find point on the line from low-high x_s(r,i) - x_s(r+1,i)
         %x_3=x1(i);
         v=((high-low)/(x_s(r+1,i)-x_s(r,i)))*(x1(i)-x_s(r,i))+low;
         end
         
         
         k=k+v;
         
         */
    }
    
    return k;
}

double IntersectionKernel_fast::calculate(arma::mat &x, int r1, arma::mat &x2, int r2){
    arma::mat combinedX=arma::join_vert(x.row(r1), x2.row(r2));
    
    arma::mat t=arma::min(combinedX);
    double r=arma::sum(arma::sum(t));

    return r;
}

float IntersectionKernel_fast::calculateKernelValue(float *x1, float *x2, int size){
    float result=0;
    
    
    // wasn't tested
    for (int i=0; i<size; i++) {
        result+=std::min(*(x1+i),*(x2+i));
    }
    
    return result;
}

/**
 *  Binary search function
 *
 *  @param x sorted column vector
 *  @param z element to search for
 *
 *  @return largest index r: x(r)<=z
 */
int IntersectionKernel_fast::binarySearch(const arma::colvec& x, double z){
    
    // Code from
    // http://stackoverflow.com/questions/20166847/faster-version-of-find-for-sorted-vectors-matlab
    int a=0;
    int b=x.size()-1;
    int c=0;
    int d=x.size()-1;
    
    while (a+1<b || c+1<d) {
        
        int mid=(int)floor((a+b)/2);
        
        if (x(mid)<z) {
            a=mid;
        }else{
            b=mid;
        }
        mid=(int)floor((c+d)/2);
        
        if (x(mid)<=z) {
            c=mid;
        }else{
            d=mid;
        }
        
        
    }
    
    //Hack. So that when x(end)< searchfor -> r=end
    if (x(b)<z) {
        c=c+1;
    }
    
    return c;
}