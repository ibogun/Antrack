//
//  SupportVector.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "SupportVector.h"


SupportVector::SupportVector(cv::Mat x_, std::vector<cv::Rect> y_,int label_,int frameIdx_){
    
    this->x=x_;
    this->y=y_;
    this->label=label_;
    this->frameIdx=frameIdx_;
    
    // create vector beta and gradient
    
    
    
    this->n=x_.cols;
    this->m=x_.rows;
    
  
    
    this->grad=cv::Mat::zeros(1, this->m, cv::DataType<float>::type);
    this->b=cv::Mat::zeros(1, this->m, cv::DataType<float>::type);

}

// C++ toString() method
std::ostream& operator<<(std::ostream &strm, const SupportVector &a) {
    
    double minVal=0;
    double maxVal=0;
    cv::minMaxIdx(a.x, &minVal, &maxVal);
    
    strm<<"Dimensions of matrix x: "<<"  [ "<<a.x.rows<<" "<<a.x.cols<<" ]\n";
    strm<<"Min Max elements of x: "<<"   [ "<<minVal<<" "<<maxVal<<" ]\n";
    
    cv::minMaxIdx(a.grad, &minVal, &maxVal);

    strm<<"Min Max elements of grad: "<<"[ "<<minVal<<" "<<maxVal<<" ]\n";
    
    cv::minMaxIdx(a.b, &minVal, &maxVal);
    
    strm<<"Min Max elements of beta: "<<"[ "<<minVal<<" "<<maxVal<<" ]\n";
    
    return strm  ;
}