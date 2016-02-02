//
// Created by Ivan Bogun on 12/3/15.
//

#include "SEEDS.h"

segmentation::SEEDS::SEEDS(int nSuperpixels, int bins, double gamma) {
    this->nSuperpixels = nSuperpixels;
    this->bins = bins;
    this->gamma = gamma;
}


void segmentation::SEEDS::initialize(const cv::Mat &image) {

    this->imageWidth = image.cols;
    this->imageHeight = image.rows;

    int rows = image.rows;
    int cols = image.cols;

    // initialize labels starting from the smallest of size 2x2

    int blockSize = smallestBlockSize;
    int numberOfSuperpixelsAtCurrentBlock = this->imageWidth *
        this->imageHeight / (blockSize * blockSize);

    while ( numberOfSuperpixelsAtCurrentBlock > this->nSuperpixels) {

        // create a label matrix and add it to the block Labels
        arma::mat blockLabels(rows,cols, arma::fill::zeros);

        int label = 1;
        for (int i = 0; i * blockSize < rows; ++i) {

            int iPixelMin = i*blockSize;
            int iPixelMax = MIN((i+1)*blockSize, rows);

            for (int j = 0; j * blockSize < cols; ++j) {
                int jPixelMin = j*blockSize;
                int jPixelMax = MIN((j+1)*blockSize, cols);

                for (int k = iPixelMin; k < iPixelMax; ++k) {    // rows
                    for (int l = jPixelMin; l < jPixelMax; ++l) {// cols
                        blockLabels(k,l) = label;
                    }
                }
                label++;
            }
        }
        allBlockLabels.push_back(blockLabels);
        blockSize = blockSize * 2;
        numberOfSuperpixelsAtCurrentBlock = this->imageWidth *
        this->imageHeight / (blockSize * blockSize);
    }
}



