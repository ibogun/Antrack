//
// Created by Ivan Bogun on 4/2/15.
//

#include "ExperimentSpatialRobustness.h"
#include "armadillo"

cv::Rect boxFromCenterAndDimensions(double x, double y, double width, double height, int n, int m) {


    if (x + width / 2.0 >= n) {
        width = n - x - 1;
    }

    if (y + height / 2.0 >= m) {
        height = m - y - 1;
    }

    int left_x = std::round(x - width / 2.0);
    int left_y = std::round(y - height / 2.0);

    if (left_x < 0) {
        left_x = 0;
    }

    if (left_y < 0) {
        left_y = 0;
    }

    cv::Rect r(left_x, left_y, std::round(width), std::round(height));

    return r;
}

std::vector<std::pair<cv::Rect, int>> ExperimentSpatialRobustness::generateBoundingBoxes(std::vector<cv::Rect> &rects,
                                                                                         int n,
                                                                                         int m) {

    std::vector<std::pair<cv::Rect, int>> boxes;

    cv::Rect gt = rects[0];// only first bounding box is necessary


    double center_x = gt.x + gt.width / 2.0;
    double center_y = gt.y + gt.height / 2.0;

    double x_shift = gt.width * this->shiftTargetSize;
    double y_shift = gt.height * this->shiftTargetSize;

    // center shifts
    std::pair<cv::Rect, int> p_top_left(
            boxFromCenterAndDimensions(center_x - x_shift, center_y - y_shift, gt.width, gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_top_right(
            boxFromCenterAndDimensions(center_x + x_shift, center_y - y_shift, gt.width, gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_bottom_left(
            boxFromCenterAndDimensions(center_x - x_shift, center_y + y_shift, gt.width, gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_bottom_right(
            boxFromCenterAndDimensions(center_x + x_shift, center_y + y_shift, gt.width, gt.height, n, m), 0);

    boxes.push_back(p_top_left);
    boxes.push_back(p_top_right);
    boxes.push_back(p_bottom_left);
    boxes.push_back(p_bottom_right);

    // corner shifts
    std::pair<cv::Rect, int> p_left(boxFromCenterAndDimensions(center_x - x_shift, center_y, gt.width, gt.height, n, m),
                                    0);
    std::pair<cv::Rect, int> p_right(
            boxFromCenterAndDimensions(center_x + x_shift, center_y, gt.width, gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_top(boxFromCenterAndDimensions(center_x, center_y - y_shift, gt.width, gt.height, n, m),
                                   0);
    std::pair<cv::Rect, int> p_bottom(
            boxFromCenterAndDimensions(center_x, center_y + y_shift, gt.width, gt.height, n, m), 0);

    boxes.push_back(p_left);
    boxes.push_back(p_right);
    boxes.push_back(p_top);
    boxes.push_back(p_bottom);

    // scale change
    std::pair<cv::Rect, int> p_smallest(
            boxFromCenterAndDimensions(center_x, center_y, pow(1 - scaleChange, 2) * gt.width,
                                       pow(1 - scaleChange, 2) * gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_smaller(
            boxFromCenterAndDimensions(center_x, center_y, pow(1 - scaleChange, 1) * gt.width,
                                       pow(1 - scaleChange, 1) * gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_bigger(boxFromCenterAndDimensions(center_x, center_y, pow(1 + scaleChange, 1) * gt.width,
                                                                 pow(1 + scaleChange, 1) * gt.height, n, m), 0);
    std::pair<cv::Rect, int> p_biggest(
            boxFromCenterAndDimensions(center_x, center_y, pow(1 + scaleChange, 2) * gt.width,
                                       pow(1 + scaleChange, 2) * gt.height, n, m), 0);

    boxes.push_back(p_smallest);
    boxes.push_back(p_smaller);
    boxes.push_back(p_bigger);
    boxes.push_back(p_biggest);

    return boxes;
}

std::ostream &operator<<(std::ostream &strm, const ExperimentSpatialRobustness &f) {
    std::string line = "--------------------------------------------------------\n";
    strm << "Experiment on spatial robustness of the tracker (SRE).\n";
    strm << "Parameters: \nscale change=" << f.scaleChange << "\nshift size=" << f.shiftTargetSize << "\n";

    return strm;
}


std::string ExperimentSpatialRobustness::getInfo() {
    std::stringstream ss;
    ss << *this;

    return ss.str();
}