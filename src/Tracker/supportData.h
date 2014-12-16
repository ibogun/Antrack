//
//  supportData.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#ifndef __Robust_tracking_by_detection__supportData__
#define __Robust_tracking_by_detection__supportData__

#include <stdio.h>
#include "armadillo"
class supportData {
    
public:
    
    arma::mat* x;
    arma::mat* y;
    
    arma::mat* beta;
    int label;
    arma::mat* grad;
    int frameNumber;
    
    
    
    supportData();
    
    //	supportData(const mat& x_,const mat& y_, const int& label_, const int& m, const int& K,int frameNumber_) {
    //		index++;
    //
    //		x = x_;
    //		y=y_;
    //		label = label_;
    //		beta = mat(1, K, fill::zeros);
    //		grad = mat(1, K, fill::zeros);
    //        frameNumber=frameNumber_;
    //
    //
    //	}
    
    //    supportData(const mat& x_,const mat& y_,const int& label_, const int&m, const int& K, int frameNumber_): x(x_),y(y_),label(label_),beta(1,K,fill::zeros),grad(1,K,fill::zeros),frameNumber(frameNumber_){
    //
    //    }
    
    
    
    supportData(const arma::mat& x_,const arma::mat& y_,const int& label_, const int&m, const int& K, int frameNumber_){
        
        x=new arma::mat(x_);
        y=new arma::mat(y_);
        beta=new arma::mat(1,K,arma::fill::zeros);
        grad=new arma::mat(1,K,arma::fill::zeros);
        label=label_;
        frameNumber=frameNumber_;
        
    }
    
    
    ~supportData(){
        delete x;
        delete y;
        delete beta;
        delete grad;
        //        delete &label;
        //        delete &frameNumber;
        
        
    }
    
};

#endif /* defined(__Robust_tracking_by_detection__supportData__) */
