//
//  Plot.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 3/19/15.
//
//

#include "Plot.h"
#include <utility>      // std::pair

void Plot::initialize(){
    
    
    const cv::Scalar AXIS_COLOR(0,0,0);
    const int thickness=1;
    /*
     1) locate the point of the origin using equation:
     origin_x=0.05*width
     origin_y=(1-0.05)*height
     */
    
    
    this->origin_x=origin_param*this->canvas.cols;
    this->origin_y=(1-origin_param)*this->canvas.rows;
    
    /*
     2) draw x and y axis
     */
    
    
    
    cv::Point origin(origin_x,origin_y);
    cv::Point topLeft(origin_x,this->canvas.rows-origin_y);
    cv::Point bottomRight(this->canvas.cols-origin_x,origin_y);
    
    cv::line(this->canvas, origin, topLeft,AXIS_COLOR,thickness);
    cv::line(this->canvas, origin, bottomRight, AXIS_COLOR,thickness);
    
    if (debug) {
        std::cout<<"Origin: "<<origin<<std::endl;
        std::cout<<"Top left: "<<topLeft<<std::endl;
        std::cout<<"Bottom right: "<<bottomRight<<std::endl;
    }
    
    std::string text="Objectness measurements";
    
    cv::Point textPoint(this->canvas.rows/2-20,20);
    cv::putText(this->canvas, text, textPoint, 1, 1, AXIS_COLOR);
    
}

void Plot::addPoint(double x,  int lineNumber){
    
    if (lines_dictionary.find(lineNumber)==lines_dictionary.end()) {
        DataPoints line;
        line.addPoint(x);
        std::pair <int,DataPoints> foo=std::make_pair(lineNumber,line);
        lines_dictionary.insert(foo);
    }else{
        
        int currentTime= lines_dictionary.at(lineNumber).size();
        
        int prevLocation_y=getYcoordinateForTime(currentTime-1);
        
        int prevLocation_x=getCoordinateForX(lines_dictionary.at(lineNumber).getLast());
        
        int currentLocation_y=getYcoordinateForTime(currentTime);
        
        int currentLocation_x=getCoordinateForX(x);
        
        cv::Point prevPoint(prevLocation_y,prevLocation_x);
        
        cv::Point currentPoint(currentLocation_y,currentLocation_x);
        
        if (debug) {
            std::cout<<"Adding line from: "<<prevPoint<<" to "<<currentPoint<<std::endl;
        }
        
        int lineNumberModThree=lineNumber%3;
        cv::Scalar lineColor(0,0,0);
        if (lineNumberModThree==0) {
            cv::Scalar c(255,255,0);
            lineColor=c;
        }else if (lineNumberModThree==2){
            cv::Scalar c(255,0,255);
            lineColor=c;
        }else{
            cv::Scalar c(100,100,255);
            lineColor=c;
            
        }
        
        
        cv::line(canvas, prevPoint, currentPoint, lineColor,2);
        lines_dictionary.at(lineNumber).addPoint(x);
        
        
    }
}




int Plot::getYcoordinateForTime(int t){
    
    if (t<0 || t>this->numPts) {
        std::cout<<"ERROR: time location exceeds maximum"<<std::endl;
        return -1;
    }else{
        cv::Point bottomRight(this->canvas.cols-origin_x,origin_y);
        
        double delta_y=(this->canvas.cols-2*canvas.cols*origin_param)/((double)numPts);
        
        int y=(int)(origin_x+delta_y*t);
        return y;
    }
}

int Plot::getCoordinateForX(double x){
    cv::Point origin(origin_x,origin_y);
    cv::Point topLeft(origin_x,this->canvas.rows-origin_y);
    
    double height=canvas.rows-2*canvas.rows*origin_param;
    
    int coordinateX=(int)(origin_y-height*x);
    
    return coordinateX;
}



//int main(int argc, const char * argv[]) {
//    
//    int pts=5;
//    Plot plot(pts);
//    plot.initialize();
//  
//    plot.addPoint(0, 1);
//    plot.addPoint(0.3, 1);
//    plot.addPoint(1, 1);
//    plot.addPoint(0.4, 1);
//    plot.addPoint(0.2, 1);
//    plot.addPoint(0.1, 1);
//    
//    int pt=3;
//    plot.addPoint(0.25, pt);
//        plot.addPoint(0.35, pt);
//        plot.addPoint(0.55, pt);
//        plot.addPoint(0.15, pt);
//    plot.show();
//   
//}
