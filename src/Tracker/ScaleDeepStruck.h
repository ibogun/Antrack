#ifndef SRC_TRACKER_SCALESTRUCK_H_
#define SRC_TRACKER_SCALESTRUCK_H_

#include "MBestStruck.h"


class ScaleStruck: public MBestStruck {

public:
        using MBestStruck::MBestStruck;

        int im_rows;
        int im_cols;
        //void initialize(cv::Mat& image)
        cv::Rect track(cv::Mat& image);

        void sampleScaleBoxes(cv::Rect& location, std::vector<cv::Rect>& rects, int num); // num*2
        void sampleTranslations(cv::Rect& location, std::vector<cv::Rect>& rects, int num, int step);// pow(num, 2)
        void sampleAspectRatio(cv::Rect& location, std::vector<cv::Rect>& rects, int num, int step);// pow(num, 2)
};



#endif
