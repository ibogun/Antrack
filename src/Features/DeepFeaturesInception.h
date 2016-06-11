// DeepFeaturesInception.h
//
// last-edit-by: <>
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifdef USE_DEEP_FEATURES
#ifndef DEEPFEATURES_H_INCEPTION
#define DEEPFEATURES_H_INCEPTION 1

#include "DeepFeatures.h"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"


class DeepFeaturesInception:public DeepFeatures {

public:
        DeepFeaturesInception(){};
        ~DeepFeaturesInception(){}

        int calculateFeatureDimension() {
                return 1024;
                //return 1000;
        }
        cv::Mat prepareImage(cv::Mat* imageIn){
                return DeepFeatures::prepareImage(imageIn);
        }
        arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
        std::string getInfo();

};


#endif
#endif // DEEPFEATURES_H_INCEPTION
//////////////////////////////////////////////////////////////////////
// $Log:$
//
