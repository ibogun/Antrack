//
//  HaarFeature.cpp
//  STR
//
//  Created by Ivan Bogun on 7/4/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "HaarFeature.h"
#include <math.h>
//typedef cv::Mat mat;
/* Given box from integral image create an instance of the HaarFeature class*/
HaarFeature::HaarFeature(cv::Mat& box,int gridLength_,int gridHeight_){
    this->integral_box=box;
    this->gridLength=gridLength_;
    this->gridHeight=gridHeight_;
}


void HaarFeature::setVariables(const cv::Mat& box,const int& gridLength_,const int& gridHeight_){
    integral_box=box;
    gridLength=gridLength_;
    gridHeight=gridHeight_;
}




/**
 *  Creates haar feature
 *
 *  @param normalize int value describing how it should be normalized (possible values: 0,1,2)
 *
 *  @return resulting feature vector
 */
arma::vec HaarFeature::calculateHaarFeature(const int normalize){

    int n=this->integral_box.rows;
    int m=this->integral_box.cols;
    
    int size=this->gridHeight*this->gridLength;
//
//    arma::rowvec two_rect_horizontal(size,arma::fill::zeros);
//    arma::rowvec two_rect_vertical(size,arma::fill::zeros);
//    
//    arma::rowvec three_rect_horizontal(size,arma::fill::zeros);
//    arma::rowvec three_rect_vertical(size,arma::fill::zeros);
//    
//    arma::rowvec four_rect_horizontal(size,arma::fill::zeros);
//    
//    arma::rowvec higher_rect_horizontal(size,arma::fill::zeros);
//    arma::rowvec higher_rect_vertical(size,arma::fill::zeros);
    
    
    arma::mat featureMatrix(7,size,arma::fill::zeros);
    

    
    /*
     Feature types are as follows:
     
     (2 - rectangular features)
     1          -       horizontal
     2          -       vertical
     (3 - rectangular features)
     3          -       horizontal
     4          -       vertical
     (4 - rectangular features)
     5          -       only one configuration exists
     ( higher order features)
     6          -       horizontal
     7          -       vertical
     */

    // Grid over the box
    std::vector<int> X=linspace(1,n-1,this->gridLength+1);
    std::vector<int> Y=linspace(1,m-1,this->gridHeight+1);
    
    cv::Mat cell;
    double t1,t2,t3,t4;
    
 
    
    //std::cout<<X.size()<<std::endl;
    //std::cout<<Y.size()<<std::endl;
    
    int counter=0;
    
    //std::cout<<"rows: "<<this->integral_box.rows<<std::endl;
    //std::cout<<"Cols: "<<this->integral_box.cols<<std::endl;
    for (int i=0; i<X.size()-1; ++i) {
        
      
        for (int j=0; j<Y.size()-1; ++j) {
            
            //std::cout<<"X[i],Y[j] "<<X[i]<<","<<Y[j]<<std::endl;
//            std::cout<<"X[i+1]-X[i] "<<X[i+1]-X[i]<<std::endl;
//            std::cout<<"Y[j+1]-Y[j] "<<Y[j+1]-Y[j]<<std::endl;

            //cv::Rect box(X[i]-1,Y[j]-1,X[i+1]-X[i]+1,Y[j+1]-Y[j]+1);
            cv::Rect box(Y[j]-1,X[i]-1,Y[j+1]-Y[j]+1,X[i+1]-X[i]+1);
//            std::cout<<box<<std::endl;
//            std::cout<<this->integral_box(box)<<std::endl;
            //std::cout<<box<<std::endl;
            cell=this->integral_box(box);
            
            int nCell=cell.rows;
            int mCell=cell.cols;
            
            int halfHorizontal=round_my(nCell/(2.0));
            int halfVertical=round_my(mCell/(2.0));
            
            int oneThirdHorizontal=round_my(nCell/(3.0));
            int oneThirdVertical=round_my(mCell/(3.0));
            
//            std::cout<<"Horizontal half "<<halfHorizontal<<std::endl;
            
            
            //         2 - rectangular features
            //1)          -       horizontal
            //std::cout<<this->integral_box<<std::endl;
            t1=getSumGivenCorners(this->integral_box,X[i]+halfHorizontal,X[i+1],Y[j],Y[j+1]);
            t2=getSumGivenCorners(this->integral_box,X[i],X[i]+halfHorizontal-1,Y[j],Y[j+1]);
            
            
           
            //two_rect_horizontal.at<double>(counter)=t1-t2;
            //two_rect_horizontal(counter)=t1-t2;

            featureMatrix(0,counter)=t1-t2;
            
            //2)          -       vertical
            t1=getSumGivenCorners(this->integral_box,X[i],X[i+1],Y[j]+halfVertical,Y[j+1]);
            t2=getSumGivenCorners(this->integral_box,X[i],X[i+1],Y[j],Y[j]+halfVertical-1);
            
            
            
            featureMatrix(1,counter)=t1-t2;
            
            //          3-rectangle features
            //3)          -       horizontal
            t1=getSumGivenCorners(this->integral_box,X[i]+oneThirdHorizontal,X[i]+2*oneThirdHorizontal-1,Y[j],Y[j+1]);
            t2=getSumGivenCorners(this->integral_box,X[i],X[i]+oneThirdHorizontal-1,Y[j],Y[j+1]);
            t3=getSumGivenCorners(this->integral_box,X[i]+2*oneThirdHorizontal,X[i+1],Y[j],Y[j+1]);
            
            
            
            
            
            //three_rect_horizontal(counter)=t1-t2-t3;
            
            featureMatrix(2,counter)=t1-t2-t3;
            
            
            
            
            //4)          -       vertical
            t1=getSumGivenCorners(this->integral_box,X[i],X[i+1],Y[j]+oneThirdVertical,Y[j]+2*oneThirdVertical-1);
            t2=getSumGivenCorners(this->integral_box,X[i],X[i+1],Y[j],Y[j]+oneThirdVertical-1);
            t3=getSumGivenCorners(this->integral_box,X[i],X[i+1],Y[j]+2*oneThirdVertical,Y[j+1]);
            
            //three_rect_vertical(counter)=t1-t2-t3;
            featureMatrix(3,counter)=t1-t2-t3;
            //            (4 - rectangular features)
            //            5          -       only one configuration exists
            t1=getSumGivenCorners(this->integral_box,X[i],X[i]+halfHorizontal-1,Y[j],Y[j]+halfVertical-1);
            t2=getSumGivenCorners(this->integral_box,X[i]+halfHorizontal,X[i+1],Y[j]+halfVertical,Y[j+1]);
            
            
            t3=getSumGivenCorners(this->integral_box,X[i]+halfHorizontal,X[i+1],Y[j],Y[j]+halfVertical-1);
            t4=getSumGivenCorners(this->integral_box,X[i],X[i]+halfHorizontal-1,Y[j]+halfVertical,Y[j+1]);

           
            
            
            
            //four_rect_horizontal(counter)=t3+t4-t1-t2;
            featureMatrix(4,counter)=t3+t4-t1-t2;


//            ( higher order features)
//            6          -       horizontal
            //higher_rect_horizontal(counter)=t1-t2;
            featureMatrix(5,counter)=t1-t2;
            
//            7          -       vertical
            //higher_rect_vertical(counter)=t3-t4;
            featureMatrix(6,counter)=t3-t4;
            
            counter++;
        }
    }
    
//    cv::Mat r1,r2,r3,r4,final;
//    hconcat(two_rect_horizontal,two_rect_vertical,r1);
//    hconcat(three_rect_horizontal,three_rect_vertical,r2);
//    
//    
//    cv::Mat r5,r6;
//    
//    hconcat(higher_rect_horizontal,higher_rect_vertical,r5);
//    
//    hconcat(r5,four_rect_horizontal,r6);
//    
//    hconcat(r6,r1,r3);
//    hconcat(r3,r2,final);
    arma::vec featureVector=arma::vectorise(featureMatrix);
    
     //std::cout<<featureMatrix<<std::endl;
    //std::cout<<featureVector<<std::endl;
    if (normalize>0) {
        
        double minVal=arma::min(arma::min(featureMatrix));
        double maxVal=arma::max(arma::max(featureMatrix));

        if (normalize==1) {
 
            // normalize to [0,1] interval
            featureVector=(featureVector-minVal)/(maxVal-minVal);
        }else{
            featureVector=((featureVector-minVal)/(maxVal-minVal)-0.5)*2;
        }
    }
   
    
    
    return featureVector;
}



inline double HaarFeature::getSumGivenCorners(cv::Mat& integral,const int& xmin,const int& xmax,const int& ymin,const int& ymax){
//
    
    // ymin->ymin-1;
    // xmin->xmin-1;
    
//    std::cout<<"Xmin="<<xmin<<" Xmax="<<xmax<<" Ymin="<<ymin<<" Ymax="<<ymax<<std::endl;
//    std::cout<<"\nPositive "<<integral.at<int>(xmax,ymax)<<"  "<<integral.at<int>(xmin-1,ymin-1)<<std::endl;
//    std::cout<<"Negative "<<integral.at<int>(xmax,ymin-1)<<"  "<<integral.at<int>(xmin-1,ymax)<<std::endl;
//    
//    
//    std::cout<<"Matlab style:"<<std::endl;
//    std::cout<<"Xmin="<<xmin<<" Xmax="<<xmax+1<<" Ymin="<<ymin<<" Ymax="<<ymax+1<<std::endl;
    //std::cout<<"\nPositive "<<integral.at<int>(xmax,ymax)<<"  "<<integral.at<int>(xmin-1,ymin-1)<<std::endl;
    //std::cout<<"Negative "<<integral.at<int>(xmax,ymin-1)<<"  "<<integral.at<int>(xmin-1,ymax)<<std::endl;
    
    
    double result=integral.at<int>(xmax,ymax)+integral.at<int>(xmin-1,ymin-1)-integral.at<int>(xmax,ymin-1)-integral.at<int>(xmin-1,ymax);

    
//    std::cout<<"Xmin="<<xmin<<" Xmax="<<xmax<<" Ymin="<<ymin<<" Ymax="<<ymax<<std::endl;
//    std::cout<<"\nPositive "<<integral.at<int>(ymax,xmax)<<"  "<<integral.at<int>(ymin-1,xmin-1)<<std::endl;
//    std::cout<<"Negative "<<integral.at<int>(ymax,xmin-1)<<"  "<<integral.at<int>(ymin-1,xmax)<<std::endl;
//    double result=integral.at<int>(ymax,xmax)+integral.at<int>(ymin-1,xmin-1)-integral.at<int>(ymax,xmin-1)-integral.at<int>(ymin-1,xmax);
////

    
    return result;
}

std::vector<int> HaarFeature::linspace(double a, double b, int n) {
    std::vector<int> array;
    double step = (b-a) / (n-1);
    
    
    while(a <= b) {
        
        array.push_back(round_my(a));
        a += step;           // could recode to better handle rounding errors
    }
    
    //array.push_back(cvRound(b));
    return array;
}


int HaarFeature::round_my(double a){
    int temp=0;
   
        if (a-floor(a)>=0.5) {
            temp=floor(a)+1;
        }else{
            temp=floor(a);
        }
    return temp;
}

