

#include "MBestStruck.h"
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include <vector>

void MBestStruck::setFeatureParams(
    const std::unordered_map<std::string, std::string> &map) {

    std::string dis_features_str = map.find("dis_features")->second;
    std::string dis_kernel_str = map.find("dis_kernel")->second;

    std::string top_features_str = map.find("top_features")->second;
    std::string top_kernel_str = map.find("top_kernel")->second;

    LOG(INFO) << "top: " << top_features_str;
    LOG(INFO) << "top: " << top_kernel_str;
    Feature *features;
    Kernel *kernel;

    if (top_features_str == "hog") {
        features = new HoG();
    } else if (top_features_str == "hist") {
        features = new HistogramFeatures(4, 16);
    } else if (top_features_str == "haar") {
        features = new Haar(2);
#ifdef USE_DEEP_FEATURES
    } else if (top_features_str == "deep") {
        features = new DeepFeatures();
        features->setParams(map);
#endif
    } else if (top_features_str == "hogANDhist") {
        Feature *f1;
        Feature *f2;
        f1 = new HistogramFeatures(4, 16);

        cv::Size winSize(32, 32);
        cv::Size blockSize(16, 16);
        cv::Size cellSize(4, 4); // was 8
        cv::Size blockStride(16, 16);
        int nBins = 8; // was 5

        f2 = new HoG(winSize, blockSize, cellSize, blockSize, nBins);

        std::vector<Feature *> mf;
        mf.push_back(f1);
        mf.push_back(f2);

        features = new MultiFeature(mf);
    } else {
        features = new RawFeatures(16);
    }

    if (top_kernel_str == "int") {
        kernel = new IntersectionKernel_fast;
    } else if (top_kernel_str == "gauss") {
        LOG(INFO) << "INITIALIZING gauss kernel";
        kernel = new RBFKernel(0.2);
    } else {
        kernel = new LinearKernel;
    }

    if (dis_features_str == "hog") {
        dis_features = new HoG();
    } else if (dis_features_str == "hist") {
        dis_features = new HistogramFeatures(4, 16);
    } else if (dis_features_str == "haar") {
        dis_features = new Haar(2);
#ifdef USE_DEEP_FEATURES
    } else if (dis_features_str == "deep") {
        dis_features = new DeepFeatures();
        dis_features->setParams(map);
#endif
    } else if (dis_features_str == "hogANDhist") {
        Feature *f1;
        Feature *f2;
        f1 = new HistogramFeatures(4, 16);

        cv::Size winSize(32, 32);
        cv::Size blockSize(16, 16);
        cv::Size cellSize(4, 4); // was 8
        cv::Size blockStride(16, 16);
        int nBins = 8; // was 5

        f2 = new HoG(winSize, blockSize, cellSize, blockSize, nBins);

        std::vector<Feature *> mf;
        mf.push_back(f1);
        mf.push_back(f2);

        dis_features = new MultiFeature(mf);
    } else {
        dis_features = new RawFeatures(16);
    }

    Kernel *dis_kernel_c;
    if (dis_kernel_str == "int") {
        dis_kernel_c = new IntersectionKernel_fast;
    } else {
        dis_kernel_c = new LinearKernel;
    }

    dis_kernel = new CachedKernel(dis_kernel_c);

    top_feature = features;
    this->top_olarank = new OLaRank_old(kernel);

    params p;
    int B = 100;
    p.C = 100;
    p.n_O = 10;
    p.n_R = 10;
    int verbose = 0;

    int m = top_feature->calculateFeatureDimension();
    this->top_olarank->setParameters(p, B, m, verbose);
}

void MBestStruck::initialize(cv::Mat &image, cv::Rect &location) {

    int m = image.rows;
    int n = image.cols;

    this->samplerForSearch->setDimensions(n, m, location.height,
                                          location.width);
    this->samplerForUpdate->setDimensions(n, m, location.height,
                                          location.width);

    this->boundingBoxes.push_back(location);
    lastLocation = location;
    // sample in polar coordinates first

    std::vector<cv::Rect> locations;

    LOG(INFO) << "Initializing regular olarank...";
    // add ground truth
    locations.push_back(location);
    this->samplerForUpdate->sampleEquiDistant(location, locations);

    cv::Mat processedImage = this->feature->prepareImage(&image);
    arma::mat x = this->feature->calculateFeature(processedImage, locations);
    arma::mat y = this->feature->reshapeYs(locations);
    this->olarank->initialize(x, y, 0, framesTracked);

    this->updateTracker = true;

    /*
      MBestTracker: initialize top detector
     */
    LOG(INFO) << "Initializing top olarank...";
    cv::Mat top_ProcessedImage = this->top_feature->prepareImage(&image);
    arma::mat top_x =
        this->top_feature->calculateFeature(top_ProcessedImage, locations);
    this->top_olarank->initialize(top_x, y, 0, framesTracked);

    // create filter, if chosen
    updateTracker = true;
    if (this->useFilter) {
        double b = 10;
        int P = 10;
        int Q = 13;
        int R = 13;
        int measurementSize = 6;
        colvec x_k(measurementSize, fill::zeros);
        x_k(0) = lastLocation.x;
        x_k(1) = lastLocation.y;
        x_k(2) = lastLocation.x + lastLocation.width;
        x_k(3) = lastLocation.y + lastLocation.height;

        int robustConstant_b = b;

        int R_cov = R;
        int Q_cov = Q;

        // FIXME: used to be this way
        //        filter =
        //        KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(
        //                x_k, n, m, R_cov, Q_cov, P, robustConstant_b);
        filter = KalmanFilterGenerator::generateConstantVelocityFilter(
            x_k, n, m, R_cov, Q_cov, P, robustConstant_b);

        filter.setBothB(robustConstant_b);
        lastRectFilterAndDetectorAgreedOn = lastLocation;

        // filter.x_kk=x_k;
    }

    // add images, in case we want to show support vectors

    if (display == 1) {
        cv::Scalar color(0, 255, 0);
        cv::Mat plotImg = image.clone();

        cv::rectangle(plotImg, lastLocation, color, 2);
        cv::imshow("Tracking window", plotImg);
        this->objectnessCanvas = plotImg;
        cv::waitKey(1);

    } else if (display == 2) {

        ObjDetectorStruck::allocateCanvas(image);
        this->frames.insert({framesTracked, image});
        this->updateDebugImage(&this->canvas, image, this->lastLocation,
                               cv::Scalar(250, 0, 0));
    } else if (display == 3) {

        // initalize here...
        // this->objPlot->initialize();
    }

    framesTracked++;

    if (pretraining) {
        this->preTraining(image, location);
    }

    // if (display == 2) {
    //     this->allocateCanvas(image);
    // }
}

cv::Rect MBestStruck::track(cv::Mat &image) {
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
        LOG(INFO) << "applied straddling";
    }

    if (useEdgeness && updateTracker && isEdgenessSane) {
        // predictions_edgeness = (predictions_edgeness -
        //                          arma::min(predictions_edgeness))/
        //                       arma::max(predictions_edgeness);
        predictions =
            predictions + this->lambda_edgeness * (predictions_edgeness);
        LOG(INFO) << "applied edgeness";
    }

    cv::Mat dis_processedImage = this->dis_features->prepareImage(&image);
    arma::mat dis_x = this->dis_features->calculateFeature(dis_processedImage,
                                                           locationsOnaGrid);

    std::vector<bool> isTopRect; // indices of the locations which are going to
                                 // the top_olarank

    std::vector<cv::Rect> topRects;
    std::vector<int> indicesOfTopVectors;

    for (int i = 0; i < locationsOnaGrid.size(); i++) {
        isTopRect.push_back(false);
    }

    LOG(INFO) << " In total there are " << locationsOnaGrid.size() << " rows";
    while (topRects.size() < this->M) {
        // find minimum over the predictions
        arma::rowvec pred_diverse(locationsOnaGrid.size(), arma::fill::zeros);

        for (int i = 0; i < locationsOnaGrid.size(); i++) {
            if (isTopRect[i]) {
                pred_diverse[i] = arma::datum::inf;
            } else {
                pred_diverse[i] = -predictions[i];
                for (int j = 0; j < topRects.size(); j++) {
                    pred_diverse[i] +=
                        this->dis_lambda *
                        (this->dis_kernel->calculate(dis_x, i, dis_x,
                                                     indicesOfTopVectors[j]));
                }
            }
        }

        uword m_best;
        pred_diverse.min(m_best);
        topRects.push_back(locationsOnaGrid[m_best]);
        indicesOfTopVectors.push_back(m_best);

        isTopRect[m_best] = true;
        LOG(INFO) << predictions[m_best] << " " << pred_diverse[m_best]
                  << locationsOnaGrid[m_best] << " id: " << m_best;
        pred_diverse[m_best] = arma::datum::inf;
    }

    //  MBestBusiness here
    // calculate features
    cv::Mat top_processedImage = this->top_feature->prepareImage(&image);
    arma::mat top_x =
        top_feature->calculateFeature(top_processedImage, topRects);

    arma::rowvec top_predictions = this->top_olarank->predictAll(top_x);

    LOG(INFO) << "Prediction scores: " << top_predictions;
    uword groundTruth;
    top_predictions.max(groundTruth);

    cv::Rect bestLocationDetector = topRects[groundTruth];

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

        LOG(INFO) << "Min / Max " << top_x_update.max() << " / "
                  << top_x_update.min();
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

typedef std::pair<double, int> my_pair;

bool comparator_mbest_pair(const my_pair &l, const my_pair &r) {
    return l.first > r.first;
}

void MBestStruck::updateDebugImage(cv::Mat *canvas, const cv::Mat &image,
                                   const std::vector<cv::Rect> &rects,
                                   const std::vector<double> &ranking,
                                   const cv::Rect &bestLocation) {

    Plot plt(rects.size());
    plt.initialize();
    for (int i = 0; i < rects.size(); i++) {
        plt.addPoint(ranking[i], 1);
        plt.next();
    }

    cv::Mat histImg = plt.getCanvas();

    cv::Scalar colorOfBox(250, 0, 0);
    int WIDTH_fitted =
        this->samplerForUpdate->objectWidth; // this dimension does not change

    int HEIGHT_fitted = this->samplerForUpdate->objectHeight;
    using namespace std;

    int imgsPerRow = 8;
    int imgsPerColumn = 8;

    int I = 0;
    int J = 0;

    cv::Vec3b color;

    // gray color
    color[0] = 128;
    color[1] = 128;
    color[2] = 128;

    for (int i = 0; i < rects.size(); i++) {

        if (I >= imgsPerRow) {
            I = 0;
            J++;
        }

        cv::Rect position(J * HEIGHT_fitted, I * WIDTH_fitted, HEIGHT_fitted,
                          WIDTH_fitted);

        cv::Mat box_img(image, rects[i]);
        cv::resize(box_img, box_img, cv::Size(WIDTH_fitted, HEIGHT_fitted));
        copyFromRectangleToImage(*canvas, box_img, position, 1, color);
        I++;
    }

    cv::Mat plotImg = image.clone();

    if (useFilter) {
        cv::rectangle(plotImg, this->lastLocationFilter,
                      cv::Scalar(255, 193, 0), 2);
    }

    for (int i = 0; i < rects.size(); i++) {
        cv::rectangle(plotImg, rects[i], cv::Scalar(0, 224, 0), 1);
    }

    cv::rectangle(plotImg, bestLocation, colorOfBox, 2);

    cv::Rect position(0, imgsPerColumn * WIDTH_fitted + 50, plotImg.rows,
                      plotImg.cols);

    color[0] = 0;
    color[1] = 0;
    color[2] = 0;
    copyFromRectangleToImage(*canvas, plotImg, position, 0, color);

    cv::resize(histImg, histImg, cv::Size(image.cols, image.rows));

    cv::Rect bottomRightRect(plotImg.rows, imgsPerColumn * WIDTH_fitted + 50,
                             histImg.rows, histImg.cols);

    this->copyFromRectangleToImage(*canvas, histImg, bottomRightRect, 0,
                                   cv::Vec3b(0, 0, 0));

    cv::Size resolution(1100, 600);

    cv::Mat finalImageToShow;

    cv::resize(*canvas, finalImageToShow, resolution);
    // cv::imshow("Support vectors", *canvas);
    cv::imshow("MBest Struck", finalImageToShow);
    cv::waitKey(1);
}

void MBestStruck::updateDebugImage(cv::Mat *canvas, cv::Mat &img,
                                   cv::Rect &bestLocation,
                                   cv::Scalar colorOfBox) {

    int WIDTH_fitted =
        this->samplerForUpdate->objectWidth; // this dimension does not change

    int HEIGHT_fitted = this->samplerForUpdate->objectHeight;
    using namespace std;
    // 1) get all support betas into the vector
    vector<my_pair> beta;
    vector<cv::Mat> bb_imgs;

    int imgsPerRow = 10;
    int imgsPerColumn = 10;

    int idx = 0;
    for (int i = 0; i < this->top_olarank->S.size(); i++) {

        int frameNumber = this->top_olarank->S[i]->frameNumber;

        //        if (frameNumber<0) {
        //            frameNumber=0;
        //        }

        cv::Mat imageToUse = this->frames.at(frameNumber);

        for (int j = 0; j < this->top_olarank->S[i]->x->n_rows; j++) {

            if ((*this->top_olarank->S[i]->beta)(j) != 0) {

                arma::mat loc = (*this->top_olarank->S[i]->y).row(j);
                cv::Rect rect(loc(1), loc(2), loc(3), loc(4));

                cv::Mat img_sp(imageToUse, rect);

                // cv::imshow("a",img_sp);
                // cv::waitKey();
                // resize image

                cv::Mat img_sp_resized(WIDTH_fitted, HEIGHT_fitted,
                                       img_sp.type());

                cv::resize(img_sp, img_sp_resized,
                           cv::Size(WIDTH_fitted, HEIGHT_fitted));
                // cv::imshow("B", img_sp_resized);
                bb_imgs.push_back(img_sp_resized);

                auto t =
                    std::make_pair((*this->top_olarank->S[i]->beta)(j), idx);
                beta.push_back(t);
                idx += 1;
            }
        }
    }

    std::sort(beta.begin(), beta.end(), comparator_mbest_pair);

    int I = 0;
    int J = 0;

    std::cout << "Total support vectors: " << beta.size() << std::endl;
    for (int i = 0; i < beta.size(); i++) {

        std::cout << "Current sv (idx,value) : " << i << " " << beta[i].first
                  << std::endl;
        if (I >= imgsPerRow) {
            I = 0;
            J++;
        }

        cv::Rect position(J * HEIGHT_fitted, I * WIDTH_fitted, HEIGHT_fitted,
                          WIDTH_fitted);

        cv::Vec3b color;
        if (beta[i].first > 0) {
            // green color
            color[1] = 255;
            color[2] = 0;
        } else {
            // red color
            color[0] = 0;
            color[1] = 0;
            color[2] = 255;
        }

        copyFromRectangleToImage(*canvas, bb_imgs[beta[i].second], position, 2,
                                 color);
        I++;
    }

    cv::Mat plotImg = img.clone();

    if (useFilter) {
        cv::rectangle(plotImg, this->lastLocationFilter,
                      cv::Scalar(255, 193, 0), 2);
    }
    cv::rectangle(plotImg, bestLocation, colorOfBox, 2);

    cv::Rect position(0, imgsPerColumn * WIDTH_fitted + 50, plotImg.rows,
                      plotImg.cols);

    cv::Vec3b color(0, 0, 0);
    copyFromRectangleToImage(*canvas, plotImg, position, 0, color);

    cv::Size resolution(1100, 600);

    cv::Mat finalImageToShow;

    cv::resize(*canvas, finalImageToShow, resolution);
    // cv::imshow("Support vectors", *canvas);
    cv::imshow("Support vectors", finalImageToShow);
    cv::waitKey(1);
    this->objectnessCanvas = finalImageToShow;
    // delete whatever is not used anymore

    for (auto it = this->frames.begin(); it != this->frames.end(); ++it) {

        //        if (it->first==0) {
        //            continue;
        //        }

        bool frameExists = false;
        for (int i = 0; i < this->top_olarank->S.size(); i++) {

            if (it->first == this->top_olarank->S[i]->frameNumber) {
                frameExists = true;
            }
        }

        if (!frameExists) {
            this->frames.erase(it->first);
        }
    }
}
