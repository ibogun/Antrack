//
//  Histogram.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Histogram.h"


cv::Mat HistogramFeatures::prepareImage(cv::Mat *imageIn){
    
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    
    if (image.channels()!=1){
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    }else{
        gray=image;
    }
    return gray;
}

arma::mat HistogramFeatures::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &locations){
    
    int numOfLevels=this->L;
    int histSize=this->NUMBER_OF_BINS_PER_HISTOGRAM;
    using namespace arma;
    // for every of the levels divide the image
    
    arma::mat x((int)locations.size(),this->calculateFeatureDimension(),arma::fill::zeros);
    
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    // for every location - extract the patch first
    for (int l=0; l<locations.size(); ++l) {
        
        // get subimage
        cv::Mat croppedImage(processedImage,locations[l]);
        
        int width=croppedImage.rows;
        int height=croppedImage.cols;
        
        // for every level L divide the image into LxL patches
        int idx=0;
        for (int L=1; L<=numOfLevels; L++) {
            vec x_rects=arma::linspace<vec>(0, width-1,L+1);
            vec y_rects=arma::linspace<vec>(0, height-1,L+1);
            
            for (int i=0; i<x_rects.size()-1; i++) {
                for (int j=0; j<y_rects.size()-1; j++) {
                    
                    // extract a patch from cropped image of size x(i)-x(i+1), y(j)-y(j+1)
                    cv::Rect roi(y_rects(j),x_rects(i),y_rects(j+1)-y_rects(j),x_rects(i+1)-x_rects(i));
                    cv::Mat patch(croppedImage,roi);
                    
                    // calculate histogram on the patch
                    
                    cv::Mat histogram;
                    cv::calcHist(&patch, 1, 0, cv::Mat(), histogram,1,&histSize,&histRange);
                    
                    float normalizationFactor=0;
                    for (int k=0; k<histSize; k++) {
                        
                        normalizationFactor+=histogram.at<float>(k);
                    }
                    
                    
                    for (int k=0; k<histSize; k++) {
                        x(l,idx)=histogram.at<float>(k)/normalizationFactor;
                        
                        idx++;
                    }
                    
                }
            }
        }
    }
    return x;
}