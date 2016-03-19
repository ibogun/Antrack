#include <stdio.h> // for snprintf
#include <string>
#include <vector>

#ifdef USE_DEEP_FEATURES
#include "DeepPCA.h"
#include <math.h>

cv::Mat DeepPCA::prepareImage(cv::Mat *imageIn) { return *imageIn; }

std::string DeepPCA::getInfo() {
    std::string result = "Features from fc7 using AlexNet";
    return result;
}

bool is_valid_double(double x) { return x * 0.0 == 0.0; }

void logMinMax(const cv::Mat &data) {
    double min, max;
    cv::minMaxLoc(data, &min, &max);

    LOG(INFO) << " Max / Min : " << max << "  " << min;
}

arma::mat DeepPCA::calculateFeature(cv::Mat &processedImage,
                                    std::vector<cv::Rect> &rects) {
    arma::mat x(static_cast<int>(rects.size()), fullFeatureSize,
                arma::fill::zeros);

    cv::Size size(this->imageSizeWidth, this->imageSizeHeight);

    std::vector<cv::Mat> images;
    std::vector<int> labels;
    for (int i = 0; i < rects.size(); i++) {
        cv::Mat roi;
        roi = processedImage(rects[i]);
        cv::Mat resizedRoi;
        cv::resize(roi, resizedRoi, size);
        images.push_back(resizedRoi);
        labels.push_back(0);
    }

    LOG(INFO) << "Started extracting deep features for " << rects.size()
              << " bounding boxes";
    boost::shared_ptr<caffe::MemoryDataLayer<float>> md_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            caffe_net->layers()[0]);

    md_layer->set_batch_size(rects.size());
    md_layer->AddMatVector(images, labels);

    float loss;
    caffe_net->ForwardPrefilled(&loss);

    boost::shared_ptr<caffe::Blob<float>> prob = caffe_net->blob_by_name("fc7");
    LOG(INFO) << "Finished extracting deep features...";
    int dim = this->fullFeatureSize;
    for (int v = 0; v < rects.size(); v++) {
        for (int i = 0; i < dim; i++) {
            x(v, i) = prob->cpu_data()[v * dim + i];
        }
    }
    // copy the data into cv::Mat
    int n = rects.size();
    int m = this->fullFeatureSize;
    cv::Mat data(n, m, CV_64F);

    int k = this->featureSize;

    if (!this->isBasisSet) {
        max_elms = arma::max(x, 0);
        min_elms = arma::min(x, 0);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if ((max_elms(j) - min_elms(j)) == 0) {
                x(i, j) = 0;
            } else {
                // x(i, j) = (x(i, j) - arma_mean(0, j)) / arma_std(0, j);
                //x(i, j) = (x(i, j) - min_elms(j)) / (max_elms(j) - min_elms(j));
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data.at<double>(i, j) = x(i, j);
        }
    }

    if (!this->isBasisSet) {
        LOG(INFO) << "Starting PCA extraction";
        pca = new cv::PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, k);
        LOG(INFO) << "Finished PCA extraction";
    }

    cv::Mat coeff(1, k, CV_64F);

    arma::mat projected_x(n, k, arma::fill::zeros);

    LOG(INFO) << "n : " << n << " m: " << m << " k: " << k;
    for (int l = 0; l < n; l++) {
        cv::Mat vec = data.row(l);
        pca->project(vec, coeff);
        for (int j = 0; j < k; j++) {

            double v = coeff.at<double>(0, j);
            if (is_valid_double(v)) {
                projected_x(l, j) = coeff.at<double>(0, j);
                CHECK_EQ(projected_x(l, j), coeff.at<double>(0, j));
            }
        }
    }

    this->isBasisSet = true;
    LOG(INFO) << "Max / Min : " << projected_x.max() << " / "
              << projected_x.min();
    LOG(INFO) << "Finished calculating projections. ";
    return projected_x;
}

void DeepPCA::setParams(
    const std::unordered_map<std::string, std::string> &map) {
    auto itProto = map.find("proto");
    auto itWeights = map.find("weights");
    auto itFeatureParams = map.find("featureSize");
    if (itProto != map.end() && itWeights != map.end()) {
        this->loadNetwork(itWeights->second, itProto->second);
    } else {
        std::cout << "To use deep features you need to specify 'proto' and "
                     "'weghts' files."
                  << "\n";
    }

    if (itFeatureParams != map.end()) {
        int i = std::stoi(itFeatureParams->second);
        this->featureSize = i;
    }
}
#endif
