//
//  HoGandRawFeatures.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/3/15.
//
//

#include "HoGandRawFeatures.h"


HoGandRawFeatures::HoGandRawFeatures(cv::Size size_,int rawSize_){
    
    rawFeatures=new RawFeatures(rawSize_);
    cv::Size winSize(64,64);
    cv::Size blockSize(32,32);
    cv::Size cellSize(8,8);
    cv::Size blockStride(16,16);
    int nBins=6;
    
    cv::HOGDescriptor* d_=new cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);
    
    this->size=size_;
    this->d=d_;
    
}

int HoGandRawFeatures::calculateFeatureDimension(){
    
    return (int)this->d->getDescriptorSize()+rawFeatures->calculateFeatureDimension();
}

cv::Mat HoGandRawFeatures::prepareImage(cv::Mat *imageIn){
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    
    if (image.channels()!=1){
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    }else{
        gray=image;
    }
    return gray;
}


std::string HoGandRawFeatures::getInfo(){
    
    std::string r="HoGandRawFeatures feature with\nwidth/height      : " +std::to_string(this->size.width)+", "+std::to_string(this->size.height)+"\n"+"Feature dimension : "+std::to_string(this->d->getDescriptorSize())+"\n";
    return r;
}

arma::mat HoGandRawFeatures::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &locationsInCropped){
    
    using namespace cv;
    
    //FIXME: Inefficient extraction of the HoG features - there should be a way to do it faster
    
   
    int n=(int)locationsInCropped.size();
    
    arma::mat x(n,(int)this->d->getDescriptorSize(),arma::fill::zeros);
    arma::mat x_raw(n,rawFeatures->calculateFeatureDimension(),arma::fill::zeros);
    
    for (int i=0; i<n; i++) {
        
        cv::Mat cropped(processedImage,locationsInCropped[i]);
        
        cv::Mat cropped_raw(processedImage,locationsInCropped[i]);
        //std::cout<<i<<" "<<locationsInCropped[i]<<std::endl;
        cv::resize(cropped,cropped,this->size);
        
        
        cv::resize(cropped_raw,cropped_raw,cv::Size(rawFeatures->size,rawFeatures->size));
        
        for (int j=0; j<rawFeatures->size; j++) {
            for (int k=0; k<rawFeatures->size; k++) {
                
                int pixel=(int)cropped_raw.at<uchar>(j,k);
                
                x_raw(i,j*rawFeatures->size+k)=(pixel*(1.0))/(255);
                
            }
        }
        
        
        std::vector<float> descriptorsValues;
        
        this->d->compute(cropped, descriptorsValues);
        
        for (int j=0; j<(int)this->d->getDescriptorSize(); j++) {
            x(i,j)=descriptorsValues[j];
        }
        
    }
    
    
    arma::mat X=arma::join_horiz(x, x_raw);
    
    
    return X;
    
    
}