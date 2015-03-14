//
//  EvaluationRun.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#ifndef __Robust_tracking_by_detection__EvaluationRun__
#define __Robust_tracking_by_detection__EvaluationRun__

#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "armadillo"

class EvaluationRun {
    
    
    int quant;

    
    friend std::ostream& operator<<(std::ostream&, const EvaluationRun&);
    
public:
    
    double precision_area = 0;
    double precision_20   = 0;

    double overlap_area   = 0;
    double overlap_half   = 0;
    
    EvaluationRun(int quant_=1000){
        this->quant=quant_;
    };
    
    void evaluate(std::vector<cv::Rect>& gt, std::vector<cv::Rect>& real);
};

#endif /* defined(__Robust_tracking_by_detection__EvaluationRun__) */
