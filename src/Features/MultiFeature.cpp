//
//  MultiFeature.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/8/15.
//
//

#include "MultiFeature.h"



cv::Mat MultiFeature::prepareImage(cv::Mat *imageIn){
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    
    if (image.channels()!=1){
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    }else{
        gray=image;
    }
    return gray;
}


int MultiFeature::calculateFeatureDimension(){
    int dim=0;
    
    for (int i=0; i<this->features.size(); i++) {
        dim+=this->features[i]->calculateFeatureDimension();
    }
    return dim;
}

std::string MultiFeature::getInfo(){
    std::string info="";
    for (int i=0; i<this->features.size(); i++) {
        
        info+=this->features[i]->getInfo()+"\n";
    }
    info+="\n";
    
    return info;
}


arma::mat MultiFeature::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &rects){
    
    
    std::vector<int> featureDimensions;
    
    std::vector<arma::mat> feature_values;
    
    for (int i=0; i<this->features.size(); i++) {

        arma::mat singleFeature=this->features[i]->calculateFeature(processedImage, rects);
        
        feature_values.push_back(singleFeature);
    }
    
    arma::mat X=feature_values[0];
    
    for (int i=1; i<this->features.size(); i++) {
        X=arma::join_horiz(X,feature_values[i]);
    }
    
    
    return X;
    
}