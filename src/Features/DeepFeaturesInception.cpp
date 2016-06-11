#include "DeepFeaturesInception.h"

arma::mat
DeepFeaturesInception::calculateFeature(cv::Mat &processedImage,
                                        std::vector<cv::Rect> &rects) {

    arma::mat x((int)rects.size(), this->calculateFeatureDimension(),
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

    std::cout << "Extracting features using Inception..."
              << "\n";
    boost::shared_ptr<caffe::MemoryDataLayer<float>> md_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            caffe_net->layers()[0]);

    md_layer->set_batch_size(rects.size());
    md_layer->AddMatVector(images, labels);

    float loss;
    caffe_net->ForwardPrefilled(&loss);

    boost::shared_ptr<caffe::Blob<float>> prob =
        caffe_net->blob_by_name("pool5/7x7_s1");

    std::cout << "been here..." << std::endl;
    int dim = this->calculateFeatureDimension();
    std::cout << "Features size: " << dim << std::endl;

    std::vector<int> sizes = prob->shape();
    for (int i = 0; i < sizes.size(); i++) {
        std::cout << sizes[i] << std::endl;
    }

    for (int v = 0; v < rects.size(); v++) {
        for (int i = 0; i < dim; i++) {
            x(v, i) = prob->cpu_data()[v * dim + i];
        }
    }

    return x;
}

std::string DeepFeaturesInception::getInfo() {
    return "DeepFeaturesInception";
};
