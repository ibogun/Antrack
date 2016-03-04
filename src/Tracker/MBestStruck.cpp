

#include "MBestStruck.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <glog/logging.h>

void MBestStruck::setFeatureParams(
    const std::unordered_map<std::string, std::string> &map) {

    std::string dis_features_str = map.find("dis_features")->second;
    std::string dis_kernel_str = map.find("dis_kernel")->second;

    std::string top_features_str = map.find("top_features")->second;
    std::string top_kernel_str = map.find("top_kernel")->second;

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

    arma::rowvec predictions = this->olarank->predictAll(x);

    // calculate features to calculate dissimilarity
    cv::Mat dis_processedImage = this->dis_features->prepareImage(&image);
    arma::mat dis_x = this->dis_features->calculateFeature(dis_processedImage,
                                                           locationsOnaGrid);

    double max_v = predictions.max();
    double min_v = predictions.min();
    predictions = (predictions - min_v) / (max_v - min_v);

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
                        this->dis_lambda/topRects.size()  *
                        (1-this->dis_kernel->calculate(dis_x, i, dis_x,
                                                     indicesOfTopVectors[j]));

                }
            }
        }

        uword m_best;
        pred_diverse.min(m_best);
        topRects.push_back(locationsOnaGrid[m_best]);
        indicesOfTopVectors.push_back(m_best);

        isTopRect[m_best] = true;
        LOG(INFO) << predictions[m_best]<< " "<< pred_diverse[m_best] << locationsOnaGrid[m_best]
                  << " id: " << m_best;
        pred_diverse[m_best] = arma::datum::inf;

    }

    cv::Mat top_processedImage = this->top_feature->prepareImage(&image);
    arma::mat top_x =
        this->top_feature->calculateFeature(top_processedImage, topRects);

    arma::rowvec top_predictions = this->top_olarank->predictAll(top_x);
    uword groundTruth;
    top_predictions.max(groundTruth);

    cv::Rect bestLocationDetector = topRects[groundTruth];

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

        arma::mat top_x_update = this->top_feature->calculateFeature(
            top_processedImage, locationsOnPolarPlane);
        this->top_olarank->process(top_x_update, y_update, 0, framesTracked);
    }
    /**
     End of tracker udpate
     **/

    if (display == 1) {
        cv::Scalar color(255, 0, 0);
        cv::Mat plotImg = image.clone();
        cv::rectangle(plotImg, lastLocation, color, 2);

        cv::imshow("Tracking window", plotImg);
        cv::waitKey(1);
        this->objectnessCanvas = plotImg;

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
    }

    framesTracked++;

    return lastLocation;
}
