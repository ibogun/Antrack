//
//  TestObjectness.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/9/15.
//
//

#ifndef __Robust_tracking_by_detection__TestObjectness__
#define __Robust_tracking_by_detection__TestObjectness__

#include "gtest/gtest.h"
#include "../Superpixels/Objectness.h"
#include <opencv2/opencv.hpp>
#include "DrawRandomImage.h"


class TestObjectness: public::testing::Test {
    
 
public:
    cv::Mat image;

    int nRects=250;
    std::vector<cv::Rect> rects;
    
    virtual void SetUp(){
        // generate random image
        DrawRandomImage draw;
        image=draw.getRandomImage();
        
        cv::RNG rng(0);

        
        for (int i=0; i<nRects; i++) {
            
            int x=rng.uniform(0, image.cols)       ;// [0;image.cols-1]
            int y=rng.uniform(0, image.rows)       ;// [0;image.rows-1]
            int w=rng.uniform(1, image.cols-x+1)   ;// [1;image.cols-x]
            int h=rng.uniform(1, image.rows-y+1)   ;// [1;image.rows-y]
            
            cv::Rect rect(x,y,w,h);
            
            if(x+w<=image.cols && y+h<=image.rows){
                rects.push_back(rect);
            }
            
        }

        int width=50;
        int height=100;
        cv::Rect r(image.cols-width,image.rows-height,width,height);
        
        rects.push_back(r);
     
    }
    
    virtual void TearDown(){
        
        image.~Mat();
    }
};

#endif /* defined(__Robust_tracking_by_detection__TestObjectness__) */
