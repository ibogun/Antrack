#include "ScaleDeepStruck.h"


bool isDimsBigEnough(cv::Rect& r) {

    if (r.width < 16) return false;
    if (r.height < 16) return false;

    return true;
}

cv::Rect ScaleStruck::track(cv::Mat &image) {
    std::vector<cv::Rect> locationsOnaGrid;
    locationsOnaGrid.push_back(lastLocation);

    this->samplerForSearch->sampleEquiDistantMultiScale(lastLocation,
                                                        locationsOnaGrid);

    if (useFilter && !updateTracker) {
        this->samplerForSearch->sampleOnAGrid(lastRectFilterAndDetectorAgreedOn,
                                              locationsOnaGrid, this->R, 2);
    }
    cv::Mat processedImage = this->feature->prepareImage(&image);
    arma::mat x =
        this->feature->calculateFeature(processedImage, locationsOnaGrid);

    LOG(INFO) << "NRects: " << locationsOnaGrid.size();

    arma::rowvec predictions = this->olarank->predictAll(x);

    arma::rowvec predictions_straddling(predictions.size(), arma::fill::zeros);
    arma::rowvec predictions_edgeness(predictions.size(), arma::fill::zeros);
    bool boxTooSmallForStraddeling = false;

    bool useEdgeness = (this->lambda_edgeness > 0);
    bool useStraddling = (this->lambda_straddeling > 0);

    bool useObjectness = (useEdgeness || useStraddling) && updateTracker;

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

        this->objectnessCanvas = cv::Mat(h, w, image.type(), CV_RGB(0, 0, 0));

        cv::Rect r(0, 0, image.rows, image.cols);
        this->copyFromRectangleToImage(this->objectnessCanvas, image, r, 2,
                                       cv::Vec3b(0, 0, 0));
    }

    Straddling *s = new Straddling(200, this->inner);
    this->straddle = *s;
    this->straddle.preprocessIntegral(small_image);
    cv::Mat gray_image;
    cv::cvtColor(small_image, gray_image, CV_RGB2GRAY);

    cv::Scalar scalar = cv::mean(gray_image);
    // EdgeDensity edge(0.1, 0.5, this->inner, this->display);
    EdgeDensity edge(0.66 * scalar[0], 1.33 * scalar[0], this->inner,
                     this->display);
    cv::Mat edges = edge.getEdges(small_image);
    edge.computeIntegrals(edges);

    cv::Rect small_image_rect(0, 0, big_box.width, big_box.height);

    int badBoxCount = 0;

    bool isStraddlingSane = true;
    bool isEdgenessSane = true;
    for (int i = 0; i < predictions.size(); ++i) {
        if (i == 0) {
            // if straddeling on the previous location of the object is
            // too small - straddeling won't help.
            double area_to_npixels =
                (small_image.rows * small_image.cols /
                 static_cast<double>(this->straddle.getNumberOfSuperpixel()));

            if (area_to_npixels * this->straddeling_threshold <=
                lastLocation.width * lastLocation.height) {
                boxTooSmallForStraddeling = true;
            }
        }

        cv::Rect rectInSmallImage(
            locationsOnaGrid[i].x - x_min, locationsOnaGrid[i].y - y_min,
            locationsOnaGrid[i].width, locationsOnaGrid[i].height);

        bool rect_fits_small_image =
            (rectInSmallImage.x + rectInSmallImage.width < small_image.cols) &&
            (rectInSmallImage.y + rectInSmallImage.height < small_image.rows) &&
            (rectInSmallImage.x >= 0) && (rectInSmallImage.y >= 0);

        if (rect_fits_small_image) {
            predictions_straddling[i] =
                this->straddle.computeStraddling(rectInSmallImage);
            if (predictions_straddling[i] != predictions_straddling[i]) {
                isStraddlingSane = false;
            }
            predictions_edgeness[i] = edge.computeEdgeDensity(rectInSmallImage);
            if (predictions_edgeness[i] != predictions_edgeness[i]) {
                isEdgenessSane = false;
            }
        } else {
            badBoxCount++;
        }

        if (this->display == 3) {
            uword maxBoxEdges;

            predictions_edgeness.max(maxBoxEdges);

            // this->edgeDensity.addToHistory(predictions_edgeness[maxBoxEdges]);

            LOG(INFO) << "Max Edgeness: " << predictions_edgeness[maxBoxEdges];
            this->objPlot->addPoint(predictions_edgeness[maxBoxEdges], 0);

            cv::Rect r = locationsOnaGrid[maxBoxEdges];

            cv::Rect bestEdgesRect(r.x - x_min, r.y - y_min, r.width, r.height);

            // convert gray to rgb
            cv::cvtColor(edges, edges, cv::COLOR_GRAY2RGB);

            // extract rectangle in the edge image
            // cv::rectangle(edges, bestEdgesRect, cv::Scalar(100, 255, 0), 2);

            // find dimensions
            int separation = 10;

            cv::Rect r3(small_image.rows, image.cols + separation,
                        small_image.rows, small_image.cols);

            this->copyFromRectangleToImage(this->objectnessCanvas, edges, r3, 2,
                                           cv::Vec3b(144, 144, 144));
            // add objectness into the this->objectness canvas
        }

        if (this->display == 3) {
            uword maxBoxStraddling;

            predictions_straddling.max(maxBoxStraddling);
            LOG(INFO) << "Max Straddling: "
                      << predictions_straddling[maxBoxStraddling];
            this->straddle.addToHistory(
                predictions_straddling[maxBoxStraddling]);

            this->objPlot->addPoint(predictions_straddling[maxBoxStraddling],
                                    1);

            cv::Rect r = locationsOnaGrid[maxBoxStraddling];
            cv::Rect bestStraddlingRect(r.x - x_min, r.y - y_min, r.width,
                                        r.height);

            // convert gray to rgb
            cv::cvtColor(straddle.canvas, straddle.canvas, cv::COLOR_GRAY2RGB);

            // cv::rectangle(straddle.canvas, bestStraddlingRect,
            // cv::Scalar(100, 255, 0),
            //              2);

            // find dimensions

            int separation = 10;
            cv::Rect r2(0, image.cols + separation, small_image.rows,
                        small_image.cols);
            this->copyFromRectangleToImage(this->objectnessCanvas,
                                           straddle.canvas, r2, 2,
                                           cv::Vec3b(144, 144, 144));
        }
    }

    predictions =
        (predictions - arma::min(predictions)) / arma::max(predictions);

    if (useStraddling && updateTracker && isStraddlingSane) {
        // predictions_straddling = (predictions_straddling -
        //                          arma::min(predictions_straddling))/
        //    arma::max(predictions_straddling);
        predictions =
            predictions + this->lambda_straddeling * predictions_straddling;
    }

    if (useEdgeness && updateTracker && isEdgenessSane) {
        // predictions_edgeness = (predictions_edgeness -
        //                          arma::min(predictions_edgeness))/
        //                       arma::max(predictions_edgeness);
        predictions =
            predictions + this->lambda_edgeness * (predictions_edgeness);
    }

    uword groundTruth;
    predictions.max(groundTruth);

    LOG(INFO) << " In total there are " << locationsOnaGrid.size() << " rows";

    cv::Rect bestLocationDetector = locationsOnaGrid[groundTruth];

    std::vector<cv::Rect> topRects;

    int scale = 8;
    int step = 2;
    int translations = 4;
    int aspect = 3;

    this->im_rows = image.rows;
    this->im_cols = image.cols;

    topRects.push_back(bestLocationDetector);
    this->sampleScaleBoxes(bestLocationDetector, topRects, scale);
    this->sampleTranslations(bestLocationDetector, topRects, translations,
                             step);
    this->sampleAspectRatio(bestLocationDetector, topRects, aspect, step);

    LOG(INFO) << " In total there are " << topRects.size() << " rows";

    //  MBestBusiness here
    // calculate features
    cv::Mat top_processedImage = this->top_feature->prepareImage(&image);
    arma::mat top_x =
        top_feature->calculateFeature(top_processedImage, topRects);

    arma::rowvec top_predictions = this->top_olarank->predictAll(top_x);

    LOG(INFO) << "Prediction scores: " << top_predictions;
    top_predictions.max(groundTruth);

    bestLocationDetector = topRects[groundTruth];

    LOG_IF(INFO, groundTruth != 0) << " Rects are different: "
                                   << "top: " << topRects[groundTruth]
                                   << " bottom: " << topRects[0];

    /**
     Filter business
     **/
    cv::Rect bestLocationFilter;
    if (useFilter) {

        arma::colvec z_k(4, arma::fill::zeros);
        z_k << bestLocationDetector.x << bestLocationDetector.y
            << bestLocationDetector.width << bestLocationDetector.height
            << endr;
        z_k(2) += z_k(0);
        z_k(3) += z_k(1);

        // make a prediction using filter
        arma::colvec x_k = filter.predict(z_k);

        bestLocationFilter = filter.getBoundingBox(
            this->lastLocation.width, this->lastLocation.height, x_k);

        this->lastLocationFilter = bestLocationFilter;

        double overlap =
            (bestLocationFilter & bestLocationDetector).area() /
            (double((bestLocationDetector | bestLocationFilter).area()));

        if (overlap > 0.5) {
            updateTracker = true;

            lastRectFilterAndDetectorAgreedOn = bestLocationDetector;
            filter.setB(filter.getGivenB());
        } else {

            updateTracker = false;

            filter.setB(filter.getGivenB() / 2.0);
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

    if (updateTracker &&
        this->boundingBoxes.size() % this->updateEveryNframes == 0) {

        // sample for updating the tracker
        std::vector<cv::Rect> locationsOnPolarPlane;
        locationsOnPolarPlane.push_back(lastLocation);

        this->samplerForUpdate->sampleEquiDistant(lastLocation,
                                                  locationsOnPolarPlane);

        // calculate features
        arma::mat x_update =
            feature->calculateFeature(processedImage, locationsOnPolarPlane);
        arma::mat y_update = this->feature->reshapeYs(locationsOnPolarPlane);
        olarank->process(x_update, y_update, 0, framesTracked);

        // calculate features
        arma::mat top_x_update = top_feature->calculateFeature(
            top_processedImage, locationsOnPolarPlane);

        arma::mat top_y_update =
            this->top_feature->reshapeYs(locationsOnPolarPlane);
        top_olarank->process(top_x_update, top_y_update, 0, framesTracked);
    }
    /**
     End of tracker udpate
     **/

    if (display == 1) {
        cv::Scalar color(255, 0, 0);
        cv::Mat plotImg = image.clone();

        cv::Mat topDetection(plotImg, lastLocation);

        cv::Rect locateDetectionRectangle(0, plotImg.cols - lastLocation.width,
                                          lastLocation.height,
                                          lastLocation.width);

        LOG(INFO) << "DEBUG: " << locateDetectionRectangle;
        this->copyFromRectangleToImage(plotImg, topDetection,
                                       locateDetectionRectangle, 0,
                                       cv::Vec3b(0, 0, 0));

        for (int i = 0; i < topRects.size(); i++) {
            cv::Scalar color(0, 0, 204);
            cv::rectangle(plotImg, topRects[i], color, 1);
        }

        if (useFilter) {
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255, 100),
                          0);
        }
        cv::rectangle(plotImg, lastLocation, color, 2);

        cv::imshow("Tracking window", plotImg);
        cv::waitKey(1);
        this->objectnessCanvas = plotImg;

    } else if (display == 2) {
        this->frames.insert({framesTracked, image});

        this->updateDebugImage(&this->canvas, image, this->lastLocation,
                               cv::Scalar(250, 0, 0));

        LOG(INFO) << lastLocation;
    } else if (display == 3) {
        // only putting the ground truth rectangle shoud be happening here
        cv::Scalar color(255, 0, 0);

        cv::Mat plotImg;
        if (useEdgeness || useStraddling) {
            plotImg = this->objectnessCanvas;
        } else {
            plotImg = image.clone();
        }

        cv::rectangle(plotImg, lastLocation, color, 2);

        if (useFilter) {

            cv::Rect bestLocationFilter = this->lastLocationFilter;
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255, 100),
                          0);
        }

        if (useEdgeness || useStraddling) {

            cv::rectangle(plotImg, lastLocationObjectness,
                          cv::Scalar(0, 204, 102), 0);

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
    delete s;
    return lastLocation;
}

void ScaleStruck::sampleScaleBoxes(cv::Rect &location,
                                   std::vector<cv::Rect> &rects, int num) {

    cv::Rect imageBox(0, 0, this->im_cols, this->im_rows);

    double scaleChange = 1.03;

    int w = location.width;
    int h = location.height;

    int w_2 = w / 2;
    int h_2 = h / 2;

    int c_x = location.x + w_2;
    int c_y = location.y + h_2;

    for (int i = -num; i <= num; i++) {
        if (i == 0)
            continue;

        double al = pow(scaleChange, i);
        int wB = cvRound(w * al);
        int hB = cvRound(h * al);

        int bb_x = c_x - wB / 2;
        int bb_y = c_y - hB / 2;

        cv::Point topLeft(bb_x, bb_y);
        cv::Point bottomRight(bb_x + wB, bb_y + hB);

        if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
            cv::Rect rect(bb_x, bb_y, wB, hB);
            if (isDimsBigEnough(rect)) {
                rects.push_back(rect);
            }
        }
    }
}

void ScaleStruck::sampleTranslations(cv::Rect &location,
                                     std::vector<cv::Rect> &rects, int num,
                                     int step) {

    cv::Rect imageBox(0, 0, this->im_cols, this->im_rows);

    int w = location.width;
    int h = location.height;

    int w_2 = w / 2;
    int h_2 = h / 2;

    int c_x = location.x + w_2;
    int c_y = location.y + h_2;

    for (int i = -num; i <= num; i++) {
        if (i == 0)
            continue;

        for (int j = -num; j <= num; j++) {

            if (j == 0)
                continue;

            int wB = w;
            int hB = h;

            int c_xB = c_x + wB / 2 - w_2 + i * step;
            int c_yB = c_y + hB / 2 - h_2 + j * step;

            int bb_x = c_xB - wB / 2;
            int bb_y = c_yB - hB / 2;

            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x + wB, bb_y + hB);

            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
                cv::Rect rect(bb_x, bb_y, wB, hB);
                if (isDimsBigEnough(rect)) {
                    rects.push_back(rect);
                }
            }
        }
    }
}

void ScaleStruck::sampleAspectRatio(cv::Rect &location,
                                    std::vector<cv::Rect> &rects, int num,
                                    int step) {

    cv::Rect imageBox(0, 0, this->im_cols, this->im_rows);
    LOG(INFO) << " SCALE STRUCK BOX: " << imageBox;
    int w = location.width;
    int h = location.height;

    int w_2 = w / 2;
    int h_2 = h / 2;

    int c_x = location.x + w_2;
    int c_y = location.y + h_2;

    for (int i = -num; i <= num; i++) {
        if (i == 0)
            continue;

        for (int j = -num; j <= num; j++) {

            if (j == 0)
                continue;

            int wB = w + (i * step);
            int hB = h + (j * step);

            int bb_x = c_x - wB / 2;
            int bb_y = c_y - hB / 2;

            cv::Point topLeft(bb_x, bb_y);
            cv::Point bottomRight(bb_x + wB, bb_y + hB);

            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
                cv::Rect rect(bb_x, bb_y, wB, hB);
                if (isDimsBigEnough(rect)) {
                    rects.push_back(rect);
                }
            }
        }
    }
}
