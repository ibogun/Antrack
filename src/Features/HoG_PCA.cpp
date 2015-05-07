//
// Created by Ivan Bogun on 4/24/15.
//

#include "HoG_PCA.h"
//
//  HoG.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/26/15.
//
//



HoG_PCA::HoG_PCA() {

    cv::Size winSize(64, 64);
    cv::Size blockSize(32, 32);
    cv::Size cellSize(8, 8); // was 8
    cv::Size blockStride(16, 16);
    int nBins = 5;          // was 5

    int k=50;

    this->winSize=winSize;
    this->blockSize=blockSize;
    this->cellSize=cellSize;
    this->blockStride=blockStride;
    this->nBins=nBins;
    this->k=k;

    cv::HOGDescriptor *d_ = new cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);

    this->d = d_;
}

int HoG_PCA::calculateFeatureDimension() {

    return this->k;
}

cv::Mat HoG_PCA::prepareImage(cv::Mat *imageIn) {
    cv::Mat image = *imageIn;
    cv::Mat gray(image.rows, image.cols, CV_16S);

    if (image.channels() != 1) {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    } else {
        gray = image;
    }
    return gray;
}


std::string HoG_PCA::getInfo() {

    std::string r = "HoG feature with\nwidth/height      : " + std::to_string(this->winSize.width) + ", " +
                    std::to_string(this->winSize.height) + "\n" + "Feature dimension : " +
                    std::to_string(this->k) +" (PCA dimension)" +"\n";
    return r;
}

arma::mat HoG_PCA::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &locationsInCropped) {

    using namespace cv;

    int m = this->d->getDescriptorSize();
    int n = (int) locationsInCropped.size();


    // find maximum and minimum x and y values from cropped

    arma::mat x(n, m, arma::fill::zeros);


    int max_x = locationsInCropped[0].x + locationsInCropped[0].width;
    int max_y = locationsInCropped[0].y + locationsInCropped[0].height;

    int min_x = locationsInCropped[0].x;
    int min_y = locationsInCropped[0].y;

    for (int i = 1; i < locationsInCropped.size(); i++) {

        min_x = MIN(min_x, locationsInCropped[i].x);
        min_y = MIN(min_y, locationsInCropped[i].y);
        max_x = MAX(max_x, locationsInCropped[i].x + locationsInCropped[i].width);
        max_y = MAX(max_y, locationsInCropped[i].y + locationsInCropped[i].height);
    }



    cv::Rect cropped_rect(min_x,min_y,max_x-min_x,max_y-min_y);
    cv::Mat cropped(processedImage,cropped_rect);

//    cv::imshow("cropped",cropped);
//    cv::waitKey();
//    cv::destroyAllWindows();

    // resize cropped image so that bounding box is the same size as the winSize of the hog


    // FIXME: Assuming that everything is done on a single scale. Won't work otherwise!

//    std::cout<<this->winSize<<std::endl;
//    std::cout<<locationsInCropped[0]<<std::endl;
    double alpha_x=((double)this->winSize.width)/(double)locationsInCropped[0].width;
    double alpha_y=((double)this->winSize.height)/(double)locationsInCropped[0].height;

    cv::resize(cropped,cropped,cv::Size(),alpha_x,alpha_y);


//    cv::imshow("cropped and resized",cropped);
//    cv::waitKey();
//    cv::destroyAllWindows();

    std::vector<cv::Point> pts;
    for (int j = 0; j < n; ++j) {

        int x=locationsInCropped[j].x-min_x;
        int y=locationsInCropped[j].y-min_y;

        // resize

        x= round(x*alpha_x);
        y=round(y*alpha_y);

        cv:Point pt(x,y);
        pts.push_back(pt);
    }

    // now calculate HoG features
    std::vector<float> descriptorValues;
    this->d->compute(cropped,descriptorValues,cv::Size(0,0),cv::Size(0,0),pts);


    cv::Mat data(n,m,CV_32FC1);

    using namespace cv;

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < m; j++) {
            //x(i, j) = descriptorValues[j+i*m];
            data.at<float>(i,j)=descriptorValues[j+i*m];
        }

    }

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, k);



    cv::Mat data_projected(n,k,CV_32FC1);

    for (int l = 0; l < n; ++l) {
        cv::Mat vec=data.row(l);
        cv::Mat coeff=data_projected.row(l);

        cv::Mat reconstructed;
        pca.project(vec,coeff);
        pca.backProject(coeff,reconstructed);



        std::cout<<vec.rows<<"  "<<vec.cols<<std::endl;
        std::cout<<coeff.rows<<"  "<<coeff.cols<<std::endl;
        std::cout<<reconstructed.rows<<"  "<<reconstructed.cols<<std::endl;
    }

    // reproject x onto the first k


    return x;


}

