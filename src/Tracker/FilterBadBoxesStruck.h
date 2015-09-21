//
// Created by Ivan Bogun on 9/18/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_FILTERBADBOXESSTRUCK_H
#define ROBUST_TRACKING_BY_DETECTION_FILTERBADBOXESSTRUCK_H

#include "Struck.h"

class FilterBadBoxesStruck:public Struck {

    public:

    FilterBadBoxesStruck(OLaRank_old* olarank_,Feature* feature_,
               LocationSampler* samplerSearch_,LocationSampler* samplerUpdate_,
               bool useObjectness_,bool scalePrior_,bool useFilter_,
               int usePretraining_,int display_):Struck(olarank_,feature_,samplerSearch_, samplerUpdate_,
    useObjectness_, scalePrior_, useFilter_,display_, usePretraining_){}

        static FilterBadBoxesStruck getTracker(bool,bool,bool,bool,bool,std::string,
                                       std::string,std::string);
        cv::Rect track(cv::Mat& image);

        cv::Rect track(std::string image_name) {
                cv::Mat image = cv::imread(image_name);
                return this->track(image);
        }

};


#endif //ROBUST_TRACKING_BY_DETECTION_FILTERBADBOXESSTRUCK_H
