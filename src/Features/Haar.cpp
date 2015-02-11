//
//  Haar.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Haar.h"
#include "HaarFeature.h"



cv::Mat Haar::prepareImage(cv::Mat *imageIn){
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    return gray;
}


std::string Haar::getInfo(){
    
    std::string result="Haar features with: "+std::to_string(dimPerScale) +" dimension \n";
    return result;
}


arma::mat Haar::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &locations){
    
    cv::Mat blurredImg;
    cv::GaussianBlur(processedImage, blurredImg, cv::Size(3,3), 0 ,0);
    
    std::vector<cv::Mat> imgs;
    imgs.push_back(blurredImg);
    
    arma::mat x((int)locations.size(),this->calculateFeatureDimension(),arma::fill::zeros);
    
    //std::cout<<x.n_rows<<" "<<x.n_cols<<std::endl;
    
    if(this->scale>1){
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
        imgs.push_back(gradientMagnitute);
    }
    int featureSize=this->dimPerScale;
    
    //        cv::Mat blurredImg2;
    //        cv::GaussianBlur(gray, blurredImg2, cv::Size(11,11), 0 ,0);
    
    
    
    for (int s=0; s<(int)this->scale; ++s) {
        // for evey scale calculate blurred image
        
        // calculate integral image
        cv::Mat integral;
        cv::integral(imgs[s], integral);
        
        
        // for every location
        for (int l=0; l<(int)locations.size(); ++l) {
            
            
            cv::Mat croppedPerLocationIntegral(integral,locations[l]);
            
            
            //arma::vec f=this->calculateHaarFeature(croppedPerLocationIntegral, 4, 4, 0);
            HaarFeature haar(croppedPerLocationIntegral, 4, 4);
            arma::vec f=haar.calculateHaarFeature(0);
            
            
            //std::cout<<f.t()<<std::endl;
            x.submat(l,s*featureSize,l,(s+1)*(featureSize)-1)=f.t();
            //std::cout<<x.row(l)<<std::endl;;
        }
        
        
    }
    
    // Do the normalization business
    
    //arma::mat mean=arma::mean(x);
    //arma::mat stddev=arma::stddev(x);
    
    int min=0;
    int max=0;
    
    for (int i=0; i<(int)locations.size(); ++i) {
        
        min=arma::min(x.row(i));
        max=arma::max(x.row(i));
        x.row(i)=((x.row(i)-min)/(max-min));
        //x.row(i)=((((x.row(i)-min)/(max-min)) - 0.5)*2);
        //x.row(i)=(x.row(i)-mean)/stddev;
        //std::cout<<x.row(i)<<std::endl;
    }
    
    
    return x;
    
}



arma::vec Haar::calculateHaarFeature(cv::Mat& integral_box,int gridHeight, int gridLength,const int normalize){
    
    int n=integral_box.rows;
    int m=integral_box.cols;
    using namespace arma;
    int size=gridHeight*gridLength;
    
    
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
    std::vector<int> X=linspace(1,n-1,gridLength+1);
    std::vector<int> Y=linspace(1,m-1,gridHeight+1);
    
    cv::Mat cell;
    double t1,t2,t3,t4;
    
    
    
    //std::cout<<X.size()<<std::endl;
    //std::cout<<Y.size()<<std::endl;
    
    int counter=0;
    
    //std::cout<<"rows: "<<this->integral_box.rows<<std::endl;
    //std::cout<<"Cols: "<<this->integral_box.cols<<std::endl;
    for (int i=0; i<(int)X.size()-1; ++i) {
        
        
        for (int j=0; j<(int)Y.size()-1; ++j) {
            //cv::Rect box(X[i]-1,Y[j]-1,X[i+1]-X[i]+1,Y[j+1]-Y[j]+1);
            cv::Rect box(Y[j]-1,X[i]-1,Y[j+1]-Y[j]+1,X[i+1]-X[i]+1);
            
            cell=integral_box(box);
            
            int nCell=cell.rows;
            int mCell=cell.cols;
            
            int halfHorizontal=round_my(nCell/(2.0));
            int halfVertical=round_my(mCell/(2.0));
            
            int oneThirdHorizontal=round_my(nCell/(3.0));
            int oneThirdVertical=round_my(mCell/(3.0));
            
            t1=getSumGivenCorners(integral_box,X[i]+halfHorizontal,X[i+1],Y[j],Y[j+1]);
            t2=getSumGivenCorners(integral_box,X[i],X[i]+halfHorizontal-1,Y[j],Y[j+1]);
            
            featureMatrix(0,counter)=t1-t2;
            
            //2)          -       vertical
            t1=getSumGivenCorners(integral_box,X[i],X[i+1],Y[j]+halfVertical,Y[j+1]);
            t2=getSumGivenCorners(integral_box,X[i],X[i+1],Y[j],Y[j]+halfVertical-1);
            
            featureMatrix(1,counter)=t1-t2;
            
            //          3-rectangle features
            //3)          -       horizontal
            t1=getSumGivenCorners(integral_box,X[i]+oneThirdHorizontal,X[i]+2*oneThirdHorizontal-1,Y[j],Y[j+1]);
            t2=getSumGivenCorners(integral_box,X[i],X[i]+oneThirdHorizontal-1,Y[j],Y[j+1]);
            t3=getSumGivenCorners(integral_box,X[i]+2*oneThirdHorizontal,X[i+1],Y[j],Y[j+1]);
            
            //three_rect_horizontal(counter)=t1-t2-t3;
            
            featureMatrix(2,counter)=t1-t2-t3;
            
            //4)          -       vertical
            t1=getSumGivenCorners(integral_box,X[i],X[i+1],Y[j]+oneThirdVertical,Y[j]+2*oneThirdVertical-1);
            t2=getSumGivenCorners(integral_box,X[i],X[i+1],Y[j],Y[j]+oneThirdVertical-1);
            t3=getSumGivenCorners(integral_box,X[i],X[i+1],Y[j]+2*oneThirdVertical,Y[j+1]);
            
            //three_rect_vertical(counter)=t1-t2-t3;
            featureMatrix(3,counter)=t1-t2-t3;
            //            (4 - rectangular features)
            //            5          -       only one configuration exists
            t1=getSumGivenCorners(integral_box,X[i],X[i]+halfHorizontal-1,Y[j],Y[j]+halfVertical-1);
            t2=getSumGivenCorners(integral_box,X[i]+halfHorizontal,X[i+1],Y[j]+halfVertical,Y[j+1]);
            
            
            t3=getSumGivenCorners(integral_box,X[i]+halfHorizontal,X[i+1],Y[j],Y[j]+halfVertical-1);
            t4=getSumGivenCorners(integral_box,X[i],X[i]+halfHorizontal-1,Y[j]+halfVertical,Y[j+1]);
            
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


std::vector<int> Haar::linspace(double a, double b, int n) {
    std::vector<int> array;
    double step = (b-a) / (n-1);
    
    
    while(a <= b) {
        
        array.push_back(round_my(a));
        a += step;           // could recode to better handle rounding errors
    }
    
    //array.push_back(cvRound(b));
    return array;
}


int Haar::round_my(double a){
    int temp=0;
    
    if (a-floor(a)>=0.5) {
        temp=floor(a)+1;
    }else{
        temp=floor(a);
    }
    return temp;
}


inline double Haar::getSumGivenCorners(cv::Mat& integral,const int& xmin,const int& xmax,const int& ymin,const int& ymax){
    
    double result=integral.at<int>(xmax,ymax)+integral.at<int>(xmin-1,ymin-1)-integral.at<int>(xmax,ymin-1)-integral.at<int>(xmin-1,ymax);
    
    return result;
}