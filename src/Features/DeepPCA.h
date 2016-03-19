// DeepFeatures.h
//
// last-edit-by: <>
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifdef USE_DEEP_FEATURES
#ifndef DEEPPCA_H
#define DEEPPCA_H 1

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


class DeepPCA:public Feature {
private:

        const int imageSizeWidth = 256;
        const int imageSizeHeight = 256;

        bool isBasisSet = false;

        int featureSize = 2048;

        cv::PCA* pca;


        const int fullFeatureSize = 4096;
        int correctedFeatureSize;

        arma::mat arma_mean;
        arma::mat arma_std;

        arma::rowvec max_elms;
        arma::rowvec min_elms;

        std::vector<int> zeroIdx;
        boost::shared_ptr<caffe::Net<float> > caffe_net;
public:
        DeepPCA() {};

        void saveMeanVariance(const cv::Mat& data);
        void normalize(cv::Mat& toNormalize);
        void computeMeanVariance(const cv::Mat& data, cv::Mat& mean, cv::Mat& std);

        ~DeepPCA() { delete pca;};


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
                return this->featureSize;
        }


};


#endif
#endif // DEEPFEATURES_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
