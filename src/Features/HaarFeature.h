//
//  HaarFeature.h
//  STR
//
//  Created by Ivan Bogun on 7/4/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __STR__HaarFeature__
#define __STR__HaarFeature__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "armadillo"
class HaarFeature{
    
private:
    cv::Mat integral_box;
    int gridHeight;
    int gridLength;
    
   
    double getSumGivenCorners(cv::Mat&,const int&,const int&,const int&,const int&);
    
    arma::vec calculateOneHaarFeature();
    
    static int round_my(double);
    
public:
    
    HaarFeature(){};
    HaarFeature(cv::Mat&,int,int);
    void setVariables(const cv::Mat&,const int&,const int&);
    static std::vector<int> linspace(double,double,int);
    arma::vec calculateHaarFeature(const int);
    
};


#endif /* defined(__STR__HaarFeature__) */
