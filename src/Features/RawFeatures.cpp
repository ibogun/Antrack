//
//  RawFeatures.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "RawFeatures.h"


/**
 *  Transforms image to the grayscale
 *
 *  @param imageIn input image
 *
 *  @return grayscale image
 */
cv::Mat RawFeatures:: prepareImage(cv::Mat *imageIn){
    
    
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    
    
    return gray;
    
}



/**
 *  Calculates Raw features for a set of rectangles
 *
 *  @param processedImage     processed image ready to be used for features extraction
 *  @param locationsInCropped vector of rectangles
 *
 *  @return matrix with normalized features
 */
arma::mat RawFeatures:: calculateFeature(cv::Mat& gray, std::vector<cv::Rect> &locationsInCropped){
    
    
    int m=this->calculateFeatureDimension();
    int n=(int)locationsInCropped.size();
    
    arma::mat x(n,m,arma::fill::zeros);
    
    for (int i=0; i<n; i++) {
        
        cv::Mat cropped(gray,locationsInCropped[i]);
        
        //std::cout<<i<<" "<<locationsInCropped[i]<<std::endl;
        cv::resize(cropped,cropped,cv::Size(size,size));
        
        for (int j=0; j<size; j++) {
            for (int k=0; k<size; k++) {
                
                int pixel=(int)cropped.at<uchar>(j,k);
                
                x(i,j*size+k)=(pixel*(1.0))/(255);
        
            }
        }
        
        
    }
    
    
    return x;
}