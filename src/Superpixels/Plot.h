//
//  Plot.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 3/19/15.
//
//

#ifndef __Robust_tracking_by_detection__Plot__
#define __Robust_tracking_by_detection__Plot__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>


class DataPoints{
    std::vector<double> pts;
    
public:
    void addPoint(double x){
        pts.push_back(x);
    }
    
    std::vector<double> getPoints(){
        
        return pts;
    }
    double getLast(){
        double last=pts.back();
        return last;
    }
    
    int size(){
        return pts.size();
    }
};

class Plot {
    int numPts;
    
    bool debug;
    
    cv::Mat canvas;
    
    int origin_x;
    int origin_y;
    
    int time;
    double const origin_param;
    
    std::unordered_map<int,DataPoints> lines_dictionary;
public:
    
    
Plot(int numPts_):debug(false),time(0),origin_param(0.08),numPts(numPts_),canvas(320,480,CV_8UC3,cv::Scalar(255,255,255)){}

    void initialize();
    
    void addPoint(double x,int lineNumber);
    
    void show(){
        cv::imshow("Plot", this->canvas);
        cv::waitKey();
        cv::destroyAllWindows();
    }
    
    void next(){
        time++;
    }
    
    int getCoordinateForX(double x);
    int getYcoordinateForTime(int t);
    
    void setNumpts(int npts){
        this->numPts=npts;
    }
    
    cv::Mat getCanvas(){
        return this->canvas;
    }
    
    //cv::Mat getCanvas(){return canvas;};
};



#endif /* defined(__Robust_tracking_by_detection__Plot__) */
