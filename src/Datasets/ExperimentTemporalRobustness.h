//
// Created by Ivan Bogun on 4/2/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_EXPERIMENTTEMPORALROBUSTNESS_H
#define ROBUST_TRACKING_BY_DETECTION_EXPERIMENTTEMPORALROBUSTNESS_H

#include "Experiment.h"

class ExperimentTemporalRobustness : public Experiment {

    int segments;

    friend std::ostream& operator<<(std::ostream&,const  ExperimentTemporalRobustness&);
public:


    ExperimentTemporalRobustness() : segments(20) { } // Wu2013 used 20 segments
    ExperimentTemporalRobustness(int segments_) : segments(segments_) { } // for user-specified number of segments
    std::vector<std::pair<cv::Rect, int>> generateBoundingBoxes(std::vector<cv::Rect> &rects, int n, int m);

    std::string getInfo();
};


#endif //ROBUST_TRACKING_BY_DETECTION_EXPERIMENTTEMPORALROBUSTNESS_H
