//
//  SupportVector.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__SupportVector__
#define __Robust_Struck__SupportVector__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>

class SupportVector {

    

    public:
    
    cv::Mat x;               // pointer to the x matrix
    cv::Mat grad;            // pointer to the gradient matrix
    std::vector<cv::Rect> y; // pointer to the y matrix
    
    int n;                  // width of the vector x
    int m;                  // height of the vector x
    
    cv::Mat b;               // pointer to the beta vector
    
    int label;              // ground truth label for the supportvector
    int frameIdx;           // frame ID for identification purposes
    

    
    
    ~SupportVector(){


    }
    SupportVector(cv::Mat x, std::vector<cv::Rect> y,int label_,int frameIdx_);
    friend std::ostream& operator<<(std::ostream&, const SupportVector&);
  
};

#endif /* defined(__Robust_Struck__SupportVector__) */
