//
// Created by Ivan Bogun on 9/18/15.
//

#ifndef SRC_TRACKER_OBJDETECTORSTRUCK_H_
#define SRC_TRACKER_OBJDETECTORSTRUCK_H_

#include <string>
#include <unordered_map>
#include "Struck.h"

class ObjDetectorStruck: public Struck {
    double lambda_straddeling = 0;
    double lambda_edgeness = 0;
    double straddeling_threshold = 0.5;
    double inner = 0.9;

 protected:
    friend std::ostream &operator<<(std::ostream &strm,
                                    const ObjDetectorStruck &s);

 public:

    using Struck::Struck;  // inherit all constructors from Struck

    cv::Rect track( cv::Mat& image);

    void setParams(const std::unordered_map<std::string, double>& map) {
        double lambda_s = map.find("lambda_straddling")->second;
        double lambda_e = map.find("lambda_edgeness")->second;
        double straddle = map.find("straddling_threshold")->second;
        double inner_param = map.find("inner")->second;

        this->inner = inner_param;
        this->setLambda(lambda_s, lambda_e);
        this->setMinStraddeling(straddle);
    }

    void setLambda(double lambda_straddling_, double lambda_edgeness_) {
        this->lambda_straddeling = lambda_straddling_;
        this->lambda_edgeness = lambda_edgeness_;
    }

    void setMinStraddeling(double straddling_threshold_) {
        this->straddeling_threshold = straddling_threshold_;
    }

    cv::Rect track(std::string image_name) {
        cv::Mat image = cv::imread(image_name);
        return this->track(image);
    }
};


#endif  // SRC_TRACKER_OBJDETECTORSTRUCK_H_
