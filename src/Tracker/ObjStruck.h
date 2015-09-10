// ObjStruck.h
//
// last-edit-by: <>
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifndef OBJSTRUCK_H
#define OBJSTRUCK_H 1

#include "Struck.h"

class ObjectStruck: public Struck {

public:

        ObjectStruck(OLaRank_old* olarank_,Feature* feature_,
               LocationSampler* samplerSearch_,LocationSampler* samplerUpdate_,
               bool useObjectness_,bool scalePrior_,bool useFilter_,
               int usePretraining_,int display_){
                olarank = olarank_;
                feature = feature_;
                samplerForSearch=samplerSearch_;
                samplerForUpdate = samplerUpdate_;
                useObjectness=useObjectness_;
                scalePrior=scalePrior_;
                useFilter=useFilter_;
                display = display_;
                pretraining=usePretraining_;
                objPlot=new Plot(500);

        };

        static ObjectStruck getTracker(bool,bool,bool,bool,bool,std::string,
                                       std::string,std::string);
        cv::Rect track(cv::Mat& image);

        cv::Rect track(std::string image_name) {
                cv::Mat image = cv::imread(image_name);
                return this->track(image);
        }
};

#endif // OBJSTRUCK_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
