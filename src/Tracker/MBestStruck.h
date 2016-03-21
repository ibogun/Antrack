// ObjStruck.h
//
// last-edit-by: <>
//
// Description:
//

//////////////////////////////////////////////////////////////////////

#ifndef SRC_TRACKER_MBESTSTRUCK_H_
#define SRC_TRACKER_MBESTSTRUCK_H_


#include "../Features/AllFeatures.h"
#include "../Kernels/CacheKernel.h"
#include "Struck.h"
#include "ObjDetectorStruck.h"
#include "OLaRank_old.h"

class MBestStruck: public ObjDetectorStruck {

public:
        OLaRank_old* top_olarank;
        Feature*  dis_features;
        CachedKernel* dis_kernel;

        Feature* top_feature;

        using ObjDetectorStruck::ObjDetectorStruck;

        int M = 64;

        double dis_lambda = 0.2;
        void setFeatureParams(const std::unordered_map<std::string, std::string> & map);

        void setM(int M_){ this->M = M_;};
        void setLambda( double dis_lambda_) {this->dis_lambda = dis_lambda_;};
        void setTopBudget(int B_) {this->top_olarank->B = B_;};
        void setBottomBudget(int B_) {this->olarank->B =B_;};
        cv::Rect track(cv::Mat& image);
        void initialize(cv::Mat& image, cv::Rect& location);

        void updateDebugImage(cv::Mat* canvas,cv::Mat& img,
                              cv::Rect &bestLocation,cv::Scalar colorOfBox);
        void updateDebugImage(cv::Mat* canvas, const cv::Mat& img,
                              const std::vector<cv::Rect>& rects, const std::vector<double>& ranking,
                              const cv::Rect& bestLocation);

        ~MBestStruck() {
                delete top_olarank;
                delete dis_features;
                delete dis_kernel;
                delete top_feature;
        }
};

#endif // OBJSTRUCK_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
