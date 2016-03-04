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
#include "OLaRank_old.h"

class MBestStruck: public Struck {

public:
        OLaRank_old* top_olarank;
        Feature*  dis_features;
        CachedKernel* dis_kernel;

        Feature* top_feature;

        using Struck::Struck;

        int M = 128;

        double dis_lambda = 2;

        void setParams(const std::unordered_map<std::string, double>& map){};
        void setFeatureParams(const std::unordered_map<std::string, std::string> & map);

        cv::Rect track(cv::Mat& image);
        void initialize(cv::Mat& image, cv::Rect& location);

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
