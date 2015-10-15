
#include <glog/logging.h>
#include <algorithm>
#include <utility>
#include <string>
#include <vector>

#include "FilterBadBoxesStruck.h"


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

cv::Rect FilterBadBoxesStruck::track(cv::Mat& image) {
    std::vector<cv::Rect> locationsOnaGrid;
    locationsOnaGrid.push_back(lastLocation);

    // Do the objectness stuff here

    int delta = this->samplerForSearch->getRadius();

    int x_min = max(0, lastLocation.x - delta);
    int y_min = max(0, lastLocation.y - delta);

    int x_max = min(image.cols, lastLocation.x + lastLocation.width + delta);
    int y_max = min(image.rows, lastLocation.y + lastLocation.height + delta);

    cv::Rect big_box(x_min, y_min, x_max - x_min, y_max - y_min);
    // extract small image from 'image'
    cv::Mat small_image(image, big_box);
    if (this->display == 3) {
            // find dimensions
            int separation = 10;
            int w = image.cols + small_image.cols + separation;
            int h = MAX(2 * image.rows, 2 * small_image.rows);

            this->objectnessCanvas = cv::Mat(h, w, image.type(), CV_RGB(0,
                                                                        0, 0));
    }

    Straddling* s  = new Straddling(200, 0.9);
    this->straddle = *s;
    this->straddle.preprocessIntegral(small_image);

    cv::Mat grey_small_image;
    cv::cvtColor(small_image, grey_small_image, CV_BGR2GRAY);
    cv::Scalar m = cv::mean(grey_small_image);

    EdgeDensity edge(0.66*m[0], 1.33*m[0], this->inner, this->display);
    edge.preprocessIntegral(grey_small_image);

    std::vector<int> radiuses;
    std::vector<int> heights;
    std::vector<int> widths;

    // populate widths and heights
    this->samplerForSearch->generateBoxesTensor(lastLocation, &radiuses,
                                                &widths, &heights);
    widths.erase(widths.begin());
    heights.erase(heights.begin());

    int c_x = lastLocation.x + lastLocation.width/2;
    int c_y = lastLocation.y + lastLocation.height/2;

    for (int i = 0; i < widths.size(); i++) {
        int width_i = widths[i];
        int height_i = heights[i];
        int top_left_x = c_x - width_i/2;
        int top_left_y = c_y - height_i/2;

        // generate bounding boxes
        cv::Rect rect_i(top_left_x, top_left_y, width_i, height_i);
        std::vector<cv::Rect> rects;
        this->samplerForSearch->sampleEquiDistant(rect_i, rects);

        std::vector<double> straddling;

        for (cv::Rect& e : rects) {
            cv::Rect rectInSmallImage(e.x-x_min, e.y - y_min , e.width,
                                      e.height);

            // check if the bounding box fits in the image
            bool rect_fits_small_image = (rectInSmallImage.x+
                                          rectInSmallImage.width <
                                          small_image.cols) &&
                (rectInSmallImage.y+ rectInSmallImage.height <
                 small_image.rows) && (rectInSmallImage.x >= 0)
                && (rectInSmallImage.y >= 0);

            double r = 0;
            if (rect_fits_small_image) {
                r = this->lambda_straddeling *
                    this->straddle.computeStraddling(rectInSmallImage);
                r+= edge.computeEdgeDensity(rectInSmallImage);
            }
            straddling.push_back(r);
        }

        std::vector<size_t> indices = sort_indexes<double>(straddling);

        for (int j = 0; j < MIN(this->topK, rects.size()); j++) {
            locationsOnaGrid.push_back(rects[rects.size() -1 -j]);
        }
    }

    delete s;

    this->samplerForSearch->sampleEquiDistant(lastLocation, locationsOnaGrid);
    std::cout << "Number of boxes: " << locationsOnaGrid.size() << std::endl;

    if (useFilter && !updateTracker) {
        this->samplerForSearch->sampleOnAGrid(lastRectFilterAndDetectorAgreedOn,
                                              locationsOnaGrid, this->R, 2);
    }

    cv::Mat processedImage = this->feature->prepareImage(&image);
    arma::mat x =
            this->feature->calculateFeature(processedImage, locationsOnaGrid);

    arma::rowvec predictions = this->olarank->predictAll(x);

    uword groundTruth;
    predictions.max(groundTruth);

    cv::Rect bestLocationDetector = locationsOnaGrid[groundTruth];

    /**
     Filter business
     **/
    cv::Rect bestLocationFilter;
    if (useFilter) {
        arma::colvec z_k(4, arma::fill::zeros);
        z_k << bestLocationDetector.x << bestLocationDetector.y
        << bestLocationDetector.width << bestLocationDetector.height << endr;
        z_k(2) += z_k(0);
        z_k(3) += z_k(1);

        // make a prediction using filter
        arma::colvec x_k = filter.predict(z_k);

        bestLocationFilter = filter.getBoundingBox(this->lastLocation.width,
                                                   this->lastLocation.height,
                                                   x_k);

        this->lastLocationFilter = bestLocationFilter;

        double overlap =
                (bestLocationFilter & bestLocationDetector).area() /
                (static_cast<double>((bestLocationDetector |
                                      bestLocationFilter).area()));

        if (overlap > 0.5) {
            updateTracker = true;
            lastRectFilterAndDetectorAgreedOn = bestLocationDetector;
            filter.setB(filter.getGivenB());
        } else {
            updateTracker = false;
            filter.setB(filter.getGivenB()/2.0);
        }
        filter.predictAndCorrect(z_k);
    }

    /**
     Final decision on the best bounding box
     **/

    // add predicted location  to the results
    this->boundingBoxes.push_back(bestLocationDetector);
    lastLocation = bestLocationDetector;

    /**
     Tracker Update
     **/

    if (updateTracker && this->boundingBoxes.size() %
        this->updateEveryNframes == 0) {

        // sample for updating the tracker
        std::vector<cv::Rect> locationsOnPolarPlane;
        locationsOnPolarPlane.push_back(lastLocation);

        this->samplerForUpdate->sampleEquiDistant(lastLocation,
                                                  locationsOnPolarPlane);

        // calculate features
        arma::mat x_update =
                feature->calculateFeature(processedImage,
                                          locationsOnPolarPlane);
        arma::mat y_update = this->feature->reshapeYs(locationsOnPolarPlane);
        olarank->process(x_update, y_update, 0, framesTracked);
    }
    /**
     End of tracker udpate
     **/

    if (display == 1) {
        cv::Scalar color(255, 0, 0);
        cv::Mat plotImg = image.clone();
        cv::rectangle(plotImg, lastLocation, color, 2);

        if (useFilter) {
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0,
                                                                  255, 100), 0);
        }

        cv::imshow("Tracking window", plotImg);
        cv::waitKey(1);

    } else if (display == 2) {
        this->frames.insert({framesTracked, image});

        this->updateDebugImage(&this->canvas, image, this->lastLocation,
                               cv::Scalar(250, 0, 0));
    } else if (display == 3) {
        // only putting the ground truth rectangle shoud be happening here
        cv::Scalar color(255, 0, 0);

        cv::Mat plotImg;
        if (this->useEdgeDensity || this->useStraddling) {
            plotImg = this->objectnessCanvas;
        } else {
            plotImg = image.clone();
        }

        cv::rectangle(plotImg, lastLocation, color, 2);

        if (useFilter) {
            cv::Rect bestLocationFilter = this->lastLocationFilter;
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255,
                                                                  100), 0);
        }


        if (this->useEdgeDensity || this->useStraddling) {
            cv::rectangle(plotImg, lastLocationObjectness, cv::Scalar(0, 204,
                                                                      102),
                          0);

            cv::Mat objPlot = this->objPlot->getCanvas();
            cv::resize(objPlot, objPlot, cv::Size(image.cols, image.rows));

            cv::Rect bottomLeftRect(image.rows + 1, 0, objPlot.rows,
                                    objPlot.cols);
            cv::rectangle(plotImg, this->gtBox, cv::Scalar(0, 0, 255), 0);
            this->copyFromRectangleToImage(plotImg, objPlot, bottomLeftRect, 0,
                                           cv::Vec3b(0, 0, 0));

            int maxWidth = 800;
            int maxHeight = 600;

            if (plotImg.rows > maxWidth || plotImg.cols > maxHeight) {
                cv::resize(plotImg, plotImg, cv::Size(maxWidth, maxHeight));
            }

            this->objectnessCanvas = plotImg;


            cv::imshow("Tracking window", plotImg);
            cv::waitKey(1);
        }
    }
    framesTracked++;
    return lastLocation;
}

