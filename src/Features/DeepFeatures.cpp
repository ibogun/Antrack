#include <stdio.h>  // for snprintf
#include <string>
#include <vector>


#ifdef USE_DEEP_FEATURES
#include "DeepFeatures.h"
cv::Mat DeepFeatures::prepareImage(cv::Mat* imageIn) {
        return *imageIn;
}

std::string DeepFeatures::getInfo() {
        std::string result = "Features from fc7 using AlexNet";
        return result;
}

arma::mat DeepFeatures::calculateFeature(cv::Mat& processedImage, std::vector<cv::Rect>& rects) {

        arma::mat x((int)rects.size(),this->calculateFeatureDimension(),arma::fill::zeros);

        cv::Size size(this->imageSizeWidth, this->imageSizeHeight);

        std::vector<cv::Mat> images;
        std::vector<int> labels;
        for (int i = 0 ; i < rects.size(); i++) {
                cv::Mat roi;
                roi = processedImage(rects[i]);
                cv::Mat resizedRoi;
                cv::resize(roi, resizedRoi, size);
                images.push_back(resizedRoi);
                labels.push_back(0);
        }

        std::cout << "Extracting features..." << "\n";
        boost::shared_ptr<caffe::MemoryDataLayer<float> > md_layer =
                boost::dynamic_pointer_cast
                <caffe::MemoryDataLayer<float> >(caffe_net->layers()[0]);

        md_layer->set_batch_size(rects.size());
        md_layer->AddMatVector(images, labels);

        float loss;
        caffe_net->ForwardPrefilled(&loss);

        boost::shared_ptr<caffe::Blob<float> > prob =
                caffe_net->blob_by_name("fc7");

        int dim = this->calculateFeatureDimension();
        for (int v = 0; v< rects.size(); v++) {
                for (int i= 0; i < dim; i++) {
                        x(v , i) = prob->cpu_data()[v*dim + i];
                }
        }

        return x;
}


void DeepFeatures::setParams(const std::unordered_map<std::string, std::string> & map) {
        auto itProto = map.find("proto");
        auto itWeights = map.find("weights");
        if ( itProto != map.end() && itWeights !=map.end()) {
                this->loadNetwork(itWeights->second, itProto->second);
        } else {
                std::cout << "To use deep features you need to specify 'proto' and 'weghts' files." << "\n";
        }

}
#endif
