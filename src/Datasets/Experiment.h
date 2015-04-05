//
//  Experiment.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#ifndef __Robust_tracking_by_detection__Experiment__
#define __Robust_tracking_by_detection__Experiment__

#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Dataset.h"


class Experiment {



public:


    virtual std::vector<std::pair<cv::Rect, int>> generateBoundingBoxes(std::vector<cv::Rect> &rects, int n, int m) = 0;
    virtual std::string getInfo()=0;

    std::vector<std::tuple<int, int, cv::Rect>> generateAllBoxesToEvaluate(Dataset *dataset);

    void showBoxes(Dataset *dataset, int vidNumber);
};

#endif /* defined(__Robust_tracking_by_detection__Experiment__) */
