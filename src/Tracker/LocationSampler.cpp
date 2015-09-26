//
//  LocationSampler.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "LocationSampler.h"
#include "algorithm"

#include <glog/logging.h>
#include <gflags/gflags.h>

/**
 *  Samples rectangles in polar coordinates
 *
 *  @param currentLocation current bounding box
 *  @param locations       vector of sampled bounding boxes
 */
void LocationSampler::sampleEquiDistant(cv::Rect &currentLocation,
                                        std::vector<cv::Rect> &locations) {

    double centerX = currentLocation.x + currentLocation.width / 2;
    double centerY = currentLocation.y + currentLocation.height / 2;

    //    std::vector<double> radialValues=linspace(0, radius, nRadial+1);
    //    std::vector<double> angularValues=linspace(0, 2*M_PI, nAngular+1);

    arma::vec radialValues = arma::linspace<arma::vec>(0, radius, nRadial + 1);
    arma::vec angularValues = arma::linspace<arma::vec>(0, 2 * M_PI, nAngular + 1);
    int bb_x, bb_y = 0;

    cv::Rect imageBox(0, 0, this->n, this->m);

    int halfWidth = cvRound(currentLocation.width / 2.0);
    int halfHeight = cvRound(currentLocation.height / 2.0);

    for (int i = 1; i < radialValues.size(); ++i) {
        for (int j = 1; j < angularValues.size(); ++j) {

            // get the top left corner
            bb_x = centerX + (radialValues(i) * cos(angularValues(j))) - halfWidth;
            bb_y = centerY + (radialValues(i) * sin(angularValues(j))) - halfHeight;

            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x + currentLocation.width, bb_y + currentLocation.height);

            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {

                cv::Rect rect(bb_x, bb_y, currentLocation.width, currentLocation.height);
                locations.push_back(rect);
            }

        }
    }
}

/**
 *  Samples rectangles in polar coordinates
 *
 *  @param currentLocation current bounding box
 *  @param locations       vector of sampled bounding boxes
 */
void LocationSampler::sampleEquiDistantMultiScale(cv::Rect &currentLocation,
                                                  std::vector<cv::Rect> &locations) {

    double centerX = currentLocation.x + currentLocation.width / 2;
    double centerY = currentLocation.y + currentLocation.height / 2;

    //    std::vector<double> radialValues=linspace(0, radius, nRadial+1);
    //    std::vector<double> angularValues=linspace(0, 2*M_PI, nAngular+1);

    arma::vec radialValues = arma::linspace<arma::vec>(0, radius, nRadial + 1);
    arma::vec angularValues = arma::linspace<arma::vec>(0, 2 * M_PI, nAngular + 1);
    int bb_x, bb_y = 0;

    cv::Rect imageBox(0, 0, this->n, this->m);

    int halfWidth = cvRound(currentLocation.width / 2.0);
    int halfHeight = cvRound(currentLocation.height / 2.0);

    for (int i = 1; i < radialValues.size(); ++i) {
        for (int j = 1; j < angularValues.size(); ++j) {

            // get the top left corner
            bb_x = centerX + (radialValues(i) * cos(angularValues(j))) - halfWidth;
            bb_y = centerY + (radialValues(i) * sin(angularValues(j))) - halfHeight;


            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x + currentLocation.width, bb_y + currentLocation.height);

            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {

                cv::Rect rect(bb_x, bb_y, currentLocation.width, currentLocation.height);
                locations.push_back(rect);
            }

        }
    }

    // CAREFUL!
    //return;



    //auto div = [](double x, double y) {return x/y;};

    int scaleR = this->radius;

    //halfWidth=cvRound(this->objectWidth/2.0);
    //halfHeight=cvRound(this->objectHeight/2.0);


    //halfWidth=cvRound(currentLocation.width/2.0);
    //halfHeight=cvRound(currentLocation.height/2.0);

    double downsample = 1.05;
    radialValues = arma::linspace<arma::vec>(0, scaleR, nRadial / 3 + 1);
    angularValues = arma::linspace<arma::vec>(0, 2 * M_PI, nAngular / 3 + 1);
    //radialValues=arma::linspace<arma::vec>(0,scaleR,3);
    //angularValues=arma::linspace<arma::vec>(0,2*M_PI, 3);

    int scale = 8;
    int shrinkOneSideScale=2;

    //return;
    std::vector<int> scale_half_width;
    std::vector<int> scale_half_height;


    //for (int scale_w=-MIN(2, scale); scale_w<=scale; scale_w++) {

    double downsample_2 = 1.05;

    for (int scale_h = -shrinkOneSideScale+1; scale_h <= shrinkOneSideScale; scale_h++) {
        int scale_w = scale_h;

        for (int scale_w = -shrinkOneSideScale+1; scale_w <= shrinkOneSideScale; scale_w++) {


            //continue;
            if ((scale_w == 0 && scale_h == 0)) {
                continue;
            }

            int halfWidth_scale = cvRound(halfWidth * pow(downsample_2, scale_w));
            int halfHeight_scale = cvRound(halfHeight * pow(downsample_2, scale_h));

            if (halfWidth_scale <= 10 || halfHeight_scale <= 10) {
                continue;
            }

            scale_half_width.push_back(halfWidth_scale);
            scale_half_height.push_back(halfHeight_scale);
        }
    }

    for (int scale_h = -scale+5; scale_h <= scale; scale_h++) {
        int scale_w = scale_h;

        if (scale_w == 0 && scale_h == 0 ) {
            continue;
        }

        int halfWidth_scale = cvRound((this->objectWidth/2.0) * pow(downsample, scale_w));
        int halfHeight_scale = cvRound((this->objectHeight/2.0) * pow(downsample, scale_h));

        if (halfWidth_scale <= 10 || halfHeight_scale <= 10) {
            continue;
        }

        scale_half_width.push_back(halfWidth_scale);
        scale_half_height.push_back(halfHeight_scale);

    }


    for (int s = 0; s < scale_half_width.size(); s++) {


        int width_scale = scale_half_width[s] * 2;
        int height_scale = scale_half_height[s] * 2;

        //double widthRatio=((double)width_scale)/this->objectWidth;
        //double heightRatio=((double)height_scale)/this->objectHeight;

        //            if (widthRatio<=0.6 || heightRatio<=0.6) {
        //                continue;
        //            }

        //            if (std::abs(div(width_scale,height_scale)-div(this->objectWidth,this->objectHeight))*(div(height_scale,width_scale)-div(this->objectHeight,this->objectWidth))>1) {
        //                continue;
        //            }

        for (int i = 0; i < radialValues.size(); ++i) {
            for (int j = 0; j < angularValues.size(); ++j) {

                // get the top left corner
                bb_x = centerX + (radialValues(i) * cos(angularValues(j))) - scale_half_width[s];
                bb_y = centerY + (radialValues(i) * sin(angularValues(j))) - scale_half_height[s];

                cv::Point topLeft(bb_x, bb_y);
                cv::Point bottomRight(bb_x + width_scale, bb_y + height_scale);

                if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {

                    cv::Rect rect(bb_x, bb_y, width_scale, height_scale);
                    locations.push_back(rect);
                }

            }
            // }

        }

    }

}


/**
 *  Sample on a grid
 *
 *  @param currentLocation current location
 *  @param locations       vector with locations
 *  @param R               radius vector to use for sampling
 *  @param step            how spread should locations be, default=1
 */
void LocationSampler::sampleOnAGrid(cv::Rect &currentLocation,
                                    std::vector<cv::Rect> &locations, int R, int step) {
    int centerX = cvRound(currentLocation.x + currentLocation.width / 2.0);
    int centerY = cvRound(currentLocation.y + currentLocation.height / 2.0);

    int halfWidth = cvRound(currentLocation.width / 2.0);
    int halfHeight = cvRound(currentLocation.height / 2.0);
    cv::Rect imageBox(0, 0, this->n, this->m);


    for (int x = -R; x <= R; x = x + step) {
        for (int y = -R; y <= R; y = y + step) {


            // make sure everything is within the radius
            if (sqrt(pow(x, 2) + pow(y, 2)) > R) {
                continue;
            }

            // get the top left corner
            int bb_x = centerX + x - halfWidth;
            int bb_y = centerY + y - halfHeight;


            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x + currentLocation.width, bb_y + currentLocation.height);

            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {

                cv::Rect rect(bb_x, bb_y, currentLocation.width, currentLocation.height);
                locations.push_back(rect);
            }

        }
    }

    int scaleR = R / 6;

    double downsample = 1.05;

    for (int scale_w = -2; scale_w <= 2; scale_w++) {

        for (int scale_h = -2; scale_h <= 2; scale_h++) {


            if (scale_w == 0 && scale_h == 0) {
                continue;
            }

            int halfWidth_scale = cvRound(halfWidth * pow(downsample, scale_w));
            int halfHeight_scale = cvRound(halfHeight * pow(downsample, scale_h));

            int width_scale = halfWidth_scale * 2;
            int height_scale = halfHeight_scale * 2;

            for (int x = -scaleR; x <= scaleR; x = x + step) {
                for (int y = -scaleR; y <= scaleR; y = y + step) {


                    // make sure everything is within the radius
                    if (sqrt(pow(x, 2) + pow(y, 2)) > scaleR) {
                        continue;
                    }

                    // get the top left corner
                    int bb_x = centerX + x - halfWidth_scale;
                    int bb_y = centerY + y - halfHeight_scale;


                    cv::Point topLeft(bb_x, bb_y);
                    cv::Point bottomRight(bb_x + width_scale, bb_y + height_scale);

                    if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {

                        cv::Rect rect(bb_x, bb_y, width_scale, height_scale);
                        locations.push_back(rect);
                    }

                }
            }
        }
    }
}


/**
 *  Samples linspace between in the interval (a,b] with n number of elements
 *
 *  @param a exclusive left bound
 *  @param b inclusive right bound
 *  @param n number of elements
 *
 *  @return vector o fthe intervals
 */
std::vector<double> LocationSampler::linspace(double a, double b, double n) {

    std::vector<double> array;
    double step = (b - a) / (n - 1);


    while (a <= b) {

        array.push_back(Haar::round_my(a));
        a += step;           // could recode to better handle rounding errors
    }

    //array.push_back(cvRound(b));
    return array;
}


inline cv::Rect LocationSampler::fromCenterToBoundingBox(const double &x, const double &y, const double &length,
                                                         const double &height) {


    // make sure that All bounding boxes are within the range
    int newX = x - length / (2.0);
    int newY = y - height / (2.0);

    int length_r = length;
    int height_r = height;

    if (newX + length > this->n - 1) {
        newX = this->n - 1 - length;
    }


    if (newY + height > this->m - 1) {
        newY = this->m - 1 - height;
    }

    if (newY < 0) {
        newY = 0;
        if (height_r > this->m - 1) {
            height_r = this->m - 1;
        }
    }
    if (newX < 0) {
        newX = 0;
        if (length_r > this->n - 1) {
            length_r = this->n - 1;
        }
    }


    cv::Rect result(newX, newY, length_r, height_r);

    return result;
}

template <typename T>
std::vector<size_t> LocationSampler::sort_indexes(const std::vector<T> &v) {
    using namespace std;
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

void LocationSampler::rearrangeByDimensionSimilarity(const cv::Rect &rect, std::vector<int> &radiuses,
                                              std::vector<int> &widths, std::vector<int> &heights) {

    std::vector<double> ratios;
    int h = rect.height;
    int w = rect.width;
    for (int i = 0; i < radiuses.size(); ++i) {
        double current_ration = sqrt(pow(widths[i]-h,2) + pow(heights[i]-h,2));
        ratios.push_back(current_ration);
    }
    std::vector<size_t > indices = sort_indexes(ratios);

    std::vector<int> radiuses_sorted;
    std::vector<int> widths_sorted;
    std::vector<int> heights_sorted;

    for (int j = 0; j < radiuses.size(); ++j) {
        radiuses_sorted.push_back(radiuses[indices[j]]);
        widths_sorted.push_back(widths[indices[j]]);
        heights_sorted.push_back(heights[indices[j]]);
    }
    radiuses =radiuses_sorted;
    widths = widths_sorted;
    heights =heights_sorted;

}

std::vector<arma::mat> LocationSampler::generateBoxesTensor(
    const int R,
    const int scale_R,
    const int min_size_half,
    const int min_scales,
    const int max_scales,
    const double downsample,
    const double shrink_one_side_scale,
    const int rows,
    const int cols,
    const cv::Rect& rect,
    std::vector<int>* radiuses,
    std::vector<int>* widths,
    std::vector<int>* heights){

    std::vector<arma::mat> objness_canvas;
    using namespace cv;

    // current height and width

    arma::mat current( rows, cols, arma::fill::zeros);
    objness_canvas.push_back(current);
    radiuses->push_back(R);
    heights->push_back(rect.height);
    widths->push_back(rect.width);

    // fixed aspect ratio
    for (int s = min_scales; s <= max_scales; s++) {
        if (s == 0) {
            continue;
        }
        int halfWidth_scale = cvRound((rect.width/2.0) *
                                      pow(downsample, s));
        int halfHeight_scale = cvRound((rect.height/2.0) *
                                       pow(downsample, s));

        if (halfWidth_scale <= min_size_half ||
            halfHeight_scale <= min_size_half ) {
            continue;
        }


        arma::mat c(rows, cols, arma::fill::zeros);

        objness_canvas.push_back(c);
        radiuses->push_back(scale_R);
        heights->push_back(halfHeight_scale*2);
        widths->push_back(halfWidth_scale * 2);

    }

    int halfWidth = rect.width/2;
    int halfHeight = rect.height/2;

    for (int scale_h = -shrink_one_side_scale+1;
         scale_h <= shrink_one_side_scale; scale_h++) {
        int scale_w = scale_h;

        for (int scale_w = -shrink_one_side_scale+1;
             scale_w <= shrink_one_side_scale; scale_w++) {
            //continue;
            if ((scale_w == 0 && scale_h == 0)) {
                continue;
            }

            int halfWidth_scale = cvRound(halfWidth * pow(downsample, scale_w));
            int halfHeight_scale = cvRound(halfHeight * pow(downsample, scale_h));

            if (halfWidth_scale <= min_size_half ||
                halfHeight_scale <= min_size_half) {
                continue;
            }

            arma::mat c(rows, cols, arma::fill::zeros);
            objness_canvas.push_back(c);
            radiuses->push_back(R);
            heights->push_back(halfHeight_scale*2);
            widths->push_back(halfWidth_scale * 2);
        }
    }


    return objness_canvas;
}

std::vector<arma::mat> LocationSampler::generateBoxesTensor(
    const cv::Rect& rect,
    std::vector<int>* radiuses,
    std::vector<int>* widths,
    std::vector<int>* heights){

    return this->generateBoxesTensor(this->radius,this->radius,
                                     this->min_size_half,this->min_scales,
                                     this->max_scales,this->downsample,
                                     this->shrink_one_side_scale,
                                     this->n, this->m,
                                     rect,
                                     radiuses,
                                     widths,
                                     heights);
}



std::ostream &operator<<(std::ostream &strm, const LocationSampler &s) {

    strm << "R                 : " << s.radius << "\n";
    strm << "nRadial           : " << s.nRadial << "\n";
    strm << "nAngular          : " << s.nAngular << "\n";
    return strm;
}
