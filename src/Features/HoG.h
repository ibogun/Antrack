//
//  HoG.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/26/15.
//
//

#ifndef __Robust_tracking_by_detection__HoG__
#define __Robust_tracking_by_detection__HoG__

#include <stdio.h>
#include "Feature.h"

class HoG : public Feature {

    cv::Size winSize;
    cv::Size blockSize;
    cv::Size cellSize;
    cv::Size blockStride;
    int nBins;

    cv::HOGDescriptor *d;
public:


    HoG(cv::Size winSize_,cv::Size blockSize_,
        cv::Size cellSize_, cv::Size blockStride_, int nBins_){

        this->winSize=winSize_;
        this->blockSize=blockSize_;
        this->cellSize=cellSize_;
        this->blockStride=blockStride_;
        this->nBins=nBins_;
        cv::HOGDescriptor *d_ = new cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);

        this->d = d_;
    };

    HoG();

    cv::Mat prepareImage(cv::Mat *imageIn);

    arma::mat calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &rects);

    int calculateFeatureDimension();


    cv::Mat get_hogdescriptor_visual_image(cv::Mat &origImg,
                                        std::vector<float> &descriptorValues,
                                        cv::Size winSize,
                                        cv::Size cellSize,
                                        int scaleFactor,
                                        double viz_factor);

    std::string getInfo();

    ~HoG() { delete d;}

};

#endif /* defined(__Robust_tracking_by_detection__HoG__) */
