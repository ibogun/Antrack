//
// Created by Ivan Bogun on 9/18/15.
//

#ifndef SRC_TRACKER_FILTERBADBOXESSTRUCK_H_
#define SRC_TRACKER_FILTERBADBOXESSTRUCK_H_

#include <string>
#include "Struck.h"

class FilterBadBoxesStruck:public Struck {
    double topK = 50;
    double lambda_straddeling = 0;
    double inner = 0.9;
 public:

    using Struck::Struck;

    void setTopK(double topK_) {
        this->topK = topK_;
    }

    void setParams(const std::unordered_map<std::string, double>& map) {
        this->setTopK(map.find("topK")->second);
        this->inner = map.find("inner")->second;
        this->lambda_straddeling = map.find("lambda_straddling")->second;
    }
        cv::Rect track(cv::Mat& image);

        cv::Rect track(std::string image_name) {
                cv::Mat image = cv::imread(image_name);
                return this->track(image);
        }
};


#endif  //  SRC_TRACKER_FILTERBADBOXESSTRUCK_H_
