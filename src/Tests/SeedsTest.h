//
// Created by Ivan Bogun on 12/8/15.
//

#ifndef SRC_TESTS_SEEDSTEST_H_
#define SRC_TESTS_SEEDSTEST_H_

#include <opencv2/opencv.hpp>

#include "gtest/gtest.h"
#include "../Superpixels/SEEDS.h"
#include "DrawRandomImage.h"


class SeedsTest: public :: testing::Test {
 public:
    segmentation::SEEDS* seeds;

    virtual  void SetUp() {
        int nSuperpixels = 100;
        int bins = 25;
        double gamma = 0;
        seeds = new segmentation::SEEDS(nSuperpixels, bins, gamma);
    }


    virtual  void TearDown() {
        delete seeds;
    }
};

#endif   //  SRC_TESTS_SEEDSTEST_H_

TEST_F(SeedsTest, testSEEDSInitialize) {
    DrawRandomImage drawRandomImage;
    cv::Mat image = drawRandomImage.getRandomImage();

    int rows = image.rows;
    int cols = image.cols;
    seeds->initialize(image);

    std::vector<arma::mat> allBlockLabels = seeds->getBlockLabels();

    int blockSize = seeds->getSmallestBlockSize();

    for (int i = 0; i < allBlockLabels.size(); ++i) {
        arma::mat blockLabel = allBlockLabels[i];
        ASSERT_EQ(blockLabel.n_rows, rows);
        ASSERT_EQ(blockLabel.n_cols, cols);

        arma::umat zeroColumns = arma::any(blockLabel == 0);
        // all pixels should have a label
        ASSERT_TRUE(arma::sum(arma::sum(zeroColumns)) == 0);

    }

}
