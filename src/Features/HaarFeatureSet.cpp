//
//  HaarFeatureSet.cpp
//  STR
//
//  Created by Ivan Bogun on 7/4/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "HaarFeatureSet.h"

#include <math.h>
#include <vector>



HaarFeatureSet::HaarFeatureSet(const cv::Mat& image_,arma::rowvec& scales_,int featureSize_,int normalization_=2){
    
    this->image=image_;
    this->scales=scales_;
    this->normalization=normalization_;
    this->featureSize=featureSize_;
}

void HaarFeatureSet::calculateFeatures(std::vector<cv::Rect>& locationsInCropped,std::vector<cv::Rect>&locations, arma::mat &x, arma::mat &y){
    
    
    /*
     1) calculate integral images at different scales
     2) allocate memory for x
     3) calculate features
     */
    
    // calculate integral images
    //int idx=0;
    
    //cv::Mat croppedIntegral;
    arma::rowvec tmp(5,arma::fill::zeros);

    cv::Mat blurredImg;
    cv::GaussianBlur(this->image, blurredImg, cv::Size(3,3), 0 ,0);
    
    
    std::vector<cv::Mat> imgs;
    imgs.push_back(blurredImg);
    
    
    // gradient image calculations follows: http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    
    /// Generate grad_x and grad_y
    cv::Mat grad_x = cv::Mat::zeros(blurredImg.rows, blurredImg.cols, CV_16S);
    cv::Mat grad_y = cv::Mat::zeros(blurredImg.rows, blurredImg.cols, CV_16S);
    

   
    /// Gradient X
    cv::Sobel(blurredImg, grad_x, CV_16S, 1, 0, 3);
    
    /// Gradient Y
    cv::Sobel(blurredImg, grad_y, CV_16S, 0, 1, 3);
    
    cv::Mat abs_grad_x, abs_grad_y;
        cv::Mat gradientMagnitute;
    
  
    
    cv::convertScaleAbs( grad_x, abs_grad_x );
    cv::convertScaleAbs( grad_y, abs_grad_y );
      cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradientMagnitute );
   

    //gradientMagnitute=grad_x+grad_y;
    
    
    
    imgs.push_back(gradientMagnitute);
    
//    cv::imshow("blurred", blurredImg);
//    cv::waitKey();
//    
//    cv::imshow("grad_x", abs_grad_x);
//    cv::waitKey();
//    
//    
//    cv::imshow("grad_y", abs_grad_y);
//    cv::waitKey();
//    
//    cv::imshow("grad_mag", gradientMagnitute);
//    cv::waitKey();
   
    for (int s=0; s<(int)this->scales.size(); ++s) {
        // for evey scale calculate blurred image
      
        
//        std::cout<<this->scales(s)<<std::endl;
//        cv::imshow("A", this->image);
//        cv::waitKey();
        
       
        
        // calculate integral image
        cv::Mat integral;
        cv::integral(imgs[s], integral);
        
        
        // for every location
        for (int l=0; l<(int)locations.size(); ++l) {
            
            cv::Mat croppedPerLocationIntegral(integral,locationsInCropped[l]);
            
            HaarFeature haar(croppedPerLocationIntegral, 4, 4);
            arma::vec f=haar.calculateHaarFeature(0);
            
            //x.row(l)=f.t();
            //x(l:l,s*(f.t().siz)
            
            //std::cout<<l<<" "<<s*this->featureSize<<" "<<l<<" "<<(s+1)*(this->featureSize-1)<<std::endl;
            x.submat(l,s*this->featureSize,l,(s+1)*(this->featureSize)-1)=f.t();
            //std::cout<<x.row(l);
            tmp<<l<<locations[l].x<<locations[l].y<<locations[l].width<<locations[l].height<<arma::endr;
            y.row(l)=tmp;
            //idx+=1;
        }
        
        
    }
    
    // Do the normalization business
    
    //arma::mat mean=arma::mean(x);
    //arma::mat stddev=arma::stddev(x);
   

    //std::cout<<mean.n_rows<<" "<<mean.n_cols<<std::endl;
    //std::cout<<stddev.n_rows<<" "<<stddev.n_cols<<std::endl;
    
    int min=0;
    int max=0;
    
    for (int i=0; i<(int)locations.size(); ++i) {
        
        min=arma::min(x.row(i));
        max=arma::max(x.row(i));
        x.row(i)=(((x.row(i)-min)/((max-min) - 0.5))*2);
        //x.row(i)=(x.row(i)-mean)/stddev;
        //std::cout<<x.row(i)<<std::endl;
    }
    
    
}