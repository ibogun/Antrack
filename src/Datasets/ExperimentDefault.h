//
// Created by Ivan Bogun on 4/2/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_EXPERIMENTDEFAULT_H
#define ROBUST_TRACKING_BY_DETECTION_EXPERIMENTDEFAULT_H
#include "Experiment.h"

class ExperimentDefault: public Experiment {

public:

    std::vector<std::pair<cv::Rect, int>> generateBoundingBoxes(std::vector<cv::Rect> &rects, int n, int m);
    std::string getInfo(){
        std::string result="Regular tracking experiment: initial bounding box per video.\n";
        return result;
    }
};


#endif //ROBUST_TRACKING_BY_DETECTION_EXPERIMENTDEFAULT_H
