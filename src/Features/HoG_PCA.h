//
// Created by Ivan Bogun on 4/24/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_HOG_PCA_H
#define ROBUST_TRACKING_BY_DETECTION_HOG_PCA_H

#include <stdio.h>
#include "Feature.h"
class HoG_PCA :public Feature{
    cv::Size winSize;
    cv::Size blockSize;
    cv::Size cellSize;
    cv::Size blockStride;
    int nBins;

    cv::HOGDescriptor *d;

    int k;
public:


    HoG_PCA(cv::Size winSize_,cv::Size blockSize_,
            cv::Size cellSize_, cv::Size blockStride_, int nBins_, int k_){

        this->winSize=winSize_;
        this->blockSize=blockSize_;
        this->cellSize=cellSize_;
        this->blockStride=blockStride_;
        this->nBins=nBins_;

        this->k=k_;
        cv::HOGDescriptor *d_ = new cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);

        this->d = d_;
    };

    HoG_PCA();

    cv::Mat prepareImage(cv::Mat *imageIn);

    arma::mat calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &rects);

    int calculateFeatureDimension();


    std::string getInfo();

    ~HoG_PCA() { delete d;}
};


#endif //ROBUST_TRACKING_BY_DETECTION_HOG_PCA_H
