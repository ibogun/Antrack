//
// Created by Ivan Bogun on 4/2/15.
//

#include "ExperimentDefault.h"

std::vector<std::pair<cv::Rect,int>> ExperimentDefault::generateBoundingBoxes(std::vector<cv::Rect>& rects, int n,
                                                                                         int m) {
    std::vector<std::pair<cv::Rect,int>> boxes;
    std::pair<cv::Rect,int> p(rects[0],0);

    boxes.push_back(p);

    return boxes;
}


