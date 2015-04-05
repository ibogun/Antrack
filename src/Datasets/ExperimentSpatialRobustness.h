//
// Created by Ivan Bogun on 4/2/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_EXPERIMENTSPATIALROBUSTNESS_H
#define ROBUST_TRACKING_BY_DETECTION_EXPERIMENTSPATIALROBUSTNESS_H

#include "Experiment.h"

class ExperimentSpatialRobustness : public Experiment {

    double scaleChange;
    double shiftTargetSize;


    friend std::ostream& operator<<(std::ostream&,const  ExperimentSpatialRobustness&);
public:

    ExperimentSpatialRobustness() :
            scaleChange(0.1),
            shiftTargetSize(0.1) { }

    ExperimentSpatialRobustness(double scaleChange_, double shiftTargetSize_) :
            scaleChange(scaleChange_),
            shiftTargetSize(shiftTargetSize_) { }

    std::vector<std::pair<cv::Rect, int>> generateBoundingBoxes(std::vector<cv::Rect> &rects, int n, int m);
    std::string getInfo();
};


#endif //ROBUST_TRACKING_BY_DETECTION_EXPERIMENTSPATIALROBUSTNESS_H
