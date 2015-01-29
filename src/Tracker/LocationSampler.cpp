//
//  LocationSampler.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "LocationSampler.h"



/**
 *  Samples rectangles in polar coordinates
 *
 *  @param currentLocation current bounding box
 *  @param locations       vector of sampled bounding boxes
 */
void LocationSampler::sampleEquiDistant(cv::Rect& currentLocation,
                                    std::vector<cv::Rect> &locations){
    
    double centerX=currentLocation.x+currentLocation.width/2;
    double centerY=currentLocation.y+currentLocation.height/2;
    
//    std::vector<double> radialValues=linspace(0, radius, nRadial+1);
//    std::vector<double> angularValues=linspace(0, 2*M_PI, nAngular+1);
    
    arma::vec radialValues=arma::linspace<arma::vec>(0,radius,nRadial+1);
    arma::vec angularValues=arma::linspace<arma::vec>(0,2*M_PI, nAngular+1);
    double deltaX,deltaY=0;
    
    for (int i=1; i<radialValues.size(); ++i) {
        for (int j=1; j<angularValues.size(); ++j) {
            
            deltaX=centerX+(radialValues(i)*cos(angularValues(j)));
            
            deltaY=centerY+(radialValues(i)*sin(angularValues(j)));
            
            locations.push_back(fromCenterToBoundingBox(deltaX,deltaY,currentLocation.width,currentLocation.height));
            
        }
    }
}


/**
 *  Sample on a grid
 *
 *  @param currentLocation current location
 *  @param locations       vector with locations
 *  @param R               radius vector to use for sampling
 *  @param step            how spread should locations be, default=1
 */
void LocationSampler::sampleOnAGrid(cv::Rect &currentLocation, std::vector<cv::Rect> &locations,int R, int step){
    int centerX=cvRound(currentLocation.x+currentLocation.width/2.0);
    int centerY=cvRound(currentLocation.y+currentLocation.height/2.0);
    
    int halfWidth=cvRound(currentLocation.width/2.0);
    int halfHeight=cvRound(currentLocation.height/2.0);
    cv::Rect imageBox(0,0,this->n,this->m);
    
    
    for (int x=-R; x<=R; x=x+step) {
        for (int y=-R; y<=R; y=y+step) {
            
            
            // make sure everything is within the radius
            if (sqrt(pow(x, 2)+pow(y, 2))>R) {
                continue;
            }
            
            // get the top left corner
            int bb_x=centerX+x-halfWidth;
            int bb_y=centerY+y-halfHeight;
            
            
            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x+currentLocation.width, bb_y+currentLocation.height);
            
            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
                
                cv::Rect rect(bb_x,bb_y,currentLocation.width,currentLocation.height);
                locations.push_back(rect);
            }
            
        }
    }
}


/**
 *  Samples linspace between in the interval (a,b] with n number of elements
 *
 *  @param a exclusive left bound
 *  @param b inclusive right bound
 *  @param n number of elements
 *
 *  @return vector o fthe intervals
 */
std::vector<double> LocationSampler::linspace(double a, double b, double n){
    
    std::vector<double> array;
    double step = (b-a) / (n-1);
    
    
    while(a <= b) {
        
        array.push_back(Haar::round_my(a));
        a += step;           // could recode to better handle rounding errors
    }
    
    //array.push_back(cvRound(b));
    return array;
}


inline cv::Rect LocationSampler::fromCenterToBoundingBox(const double& x,const double& y,const double& length,const double& height){
    
    
    // make sure that All bounding boxes are within the range
    int newX=x-length/(2.0);
    int newY=y-height/(2.0);
    
    int length_r=length;
    int height_r=height;
    
    if (newX+length>this->n-1){
        newX=this->n-1-length;
    }
    
    
    if (newY+height>this->m-1){
        newY=this->m-1-height;
    }
    
    if (newY<0){
        newY=0;
        if (height_r>this->m-1) {
            height_r=this->m-1;
        }
    }
    if (newX<0){
        newX=0;
        if (length_r>this->n-1) {
            length_r=this->n-1;
        }
    }
    
    
    cv::Rect result(newX,newY,length_r,height_r);
    
    return result;
}


std::ostream& operator<<(std::ostream& strm, const LocationSampler& s){
    
    strm<<"R                 : "<<s.radius<<"\n";
    strm<<"nRadial           : "<<s.nRadial<<"\n";
    strm<<"nAngular          : "<<s.nAngular<<"\n";
    return strm;
}