// DeepFeatures.h
//
// last-edit-by: <>
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifdef USE_DEEP_FEATURES
#ifndef DEEPFEATURES_H
#define DEEPFEATURES_H 1

#include "Feature.h"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"


class DeepFeatures:public Feature {
private:

        const int imageSizeWidth = 256;
        const int imageSizeHeight = 256;

        boost::shared_ptr<caffe::Net<float> > caffe_net;
public:
        DeepFeatures() {};


        void setParams (const std::unordered_map<std::string, std::string> & map );

        void loadNetwork(std::string weight_file_,
                         std::string network_definition_file_) {

                std::string pretrained_binary_proto(weight_file_);
                std::string feature_extraction_proto(network_definition_file_);
                boost::shared_ptr<caffe::Net<float> > caffe_net(
                        new caffe::Net<float>(feature_extraction_proto, caffe::TEST));
                caffe_net->CopyTrainedLayersFrom(pretrained_binary_proto);

                this->caffe_net = caffe_net;
        }
        cv::Mat prepareImage(cv::Mat* imageIn);
        arma::mat calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects);
        std::string getInfo();

        int calculateFeatureDimension() {
                return 4096;
        }


};


#endif
#endif // DEEPFEATURES_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
