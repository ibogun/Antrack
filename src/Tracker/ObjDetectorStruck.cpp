//
// Created by Ivan Bogun on 9/18/15.
//

#include "ObjDetectorStruck.h"
#include  <glog/logging.h>

bool isBBoxTooSmallForStraddeling(cv::Rect& rect, int area,
                                  int nSuperpixels, double threshold){

    int box_area = rect.width * rect.height;

    return (threshold*(area/((double) nSuperpixels)) < box_area);
}

cv::Rect ObjDetectorStruck::track(cv::Mat& image){

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
    arma::rowvec obj_predictions(predictions.size(), arma::fill::zeros);
    bool boxTooSmallForStraddeling=false;

    if (this->lambda > 0 && updateTracker){
        int delta = this->samplerForSearch->getRadius();
    
        int x_min = max(0, lastLocation.x - delta);
        int y_min = max(0, lastLocation.y - delta);
    
        int x_max = min(image.cols, lastLocation.x +
                        lastLocation.width + delta);
        int y_max = min(image.rows, lastLocation.y +
                        lastLocation.height + delta);
    
        cv::Rect big_box(x_min, y_min, x_max - x_min, y_max - y_min);
        // extract small image from 'image'
        cv::Mat small_image(image, big_box);

        Straddling* s  = new Straddling(200, 0.9);
        this->straddle = *s;
        this->straddle.preprocessIntegral(small_image);

        cv::Rect small_image_rect(0,0, big_box.width, big_box.height);

        for (int i = 0; i < predictions.size(); ++i) {

            if (i == 0){

                // if straddeling on the previous location of the object is
                // too small - straddeling won't help.
                double area_to_npixels =(small_image.rows*small_image.cols/
                                         (double)this->straddle.getNumberOfSuperpixel());

                LOG(INFO)<<"Min area in pixels: " << area_to_npixels*this->straddeling_threshold;
                LOG(INFO)<<"Area of the box: "  << lastLocation.width * lastLocation.height;
                LOG(INFO) << "Straddling is: " << (area_to_npixels*this->straddeling_threshold
                    <= lastLocation.width * lastLocation.height);
                if (area_to_npixels*this->straddeling_threshold
                    <= lastLocation.width * lastLocation.height) {
                    boxTooSmallForStraddeling= true;
                    break;
                }
            }

            cv::Rect rectInSmallImage(locationsOnaGrid[i].x - x_min,
                                      locationsOnaGrid[i].y - y_min,
                                      locationsOnaGrid[i].width,
                                      locationsOnaGrid[i].height);

            bool rect_fits_small_image = (rectInSmallImage.x+
                                          rectInSmallImage.width <
                                          small_image.cols) &&
                (rectInSmallImage.y+ rectInSmallImage.height <
                 small_image.rows)&& (rectInSmallImage.x >= 0)
                && ( rectInSmallImage.y >= 0);

            if (rect_fits_small_image){
                obj_predictions[i] = this->straddle.computeStraddling(
                    rectInSmallImage);
            }
        }

        delete s;
    }



    if (lambda > 0 && !boxTooSmallForStraddeling && updateTracker)  {
        predictions = (predictions - arma::min(predictions))/
            arma::max(predictions);
        obj_predictions = (obj_predictions - arma::min(obj_predictions))/
            arma::max(obj_predictions);
        predictions = predictions + lambda* obj_predictions;
    }

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
                (double((bestLocationDetector | bestLocationFilter).area()));

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
        this->updateEveryNframes==0) {

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
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255,
                                                                  100), 0);
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
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255, 100),
                          0);
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

ObjDetectorStruck ObjDetectorStruck::getTracker(bool pretraining,
                                                bool useFilter,
                                                bool useEdgeDensity,
                                                bool useStraddling,
                                                bool scalePrior,
                                                std::string kernelSTR,
                                                std::string featureSTR,
                                                std::string note_) {

    // Parameters
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params p;
    p.C = 100;
    p.n_O = 10;
    p.n_R = 10;
    int nRadial = 5;
    int nAngular = 16;
    int B = 100;

    int nRadial_search = 12;
    int nAngular_search = 30;

    // RawFeatures* features=new RawFeatures(16);


    Feature *features;
    Kernel *kernel;

    if (featureSTR == "hog") {
        features = new HoG();
    } else if (featureSTR == "hist") {
        features = new HistogramFeatures(4, 16);
    } else if (featureSTR == "haar"){
        features = new Haar(2);
    } else if (featureSTR == "hogANDhist"){
        Feature* f1;
        Feature* f2;
        f1=new HistogramFeatures(4,16);
        cv::Size winSize(32, 32);
        cv::Size blockSize(16, 16);
        cv::Size cellSize(4, 4); // was 8
        cv::Size blockStride(16, 16);
        int nBins = 8;          // was 5

        f2= new HoG(winSize,blockSize,cellSize,blockSize,nBins);

        std::vector<Feature*> mf;
        mf.push_back(f1);
        mf.push_back(f2);
        features = new MultiFeature(mf);
    }

    else {
        features = new RawFeatures(16);
    }

    if (kernelSTR == "int") {
        kernel = new IntersectionKernel_fast;
    } else if (kernelSTR == "gauss") {
        kernel = new RBFKernel(0.2);
    }else if (kernelSTR == "approxGauss") {

        RBFKernel* rbf=new RBFKernel(0.2);
        int pts=25;
        kernel = new ApproximateKernel(pts,rbf);
    } else {
        kernel = new LinearKernel;
    }

    int verbose = 0;
    int display = 0;
    int m = features->calculateFeatureDimension();

    OLaRank_old *olarank = new OLaRank_old(kernel);
    olarank->setParameters(p, B, m, verbose);

    int r_search = 45;
    int r_update = 60;

    bool useObjectness = (useEdgeDensity | useStraddling);


    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    LocationSampler *samplerForUpdate =
            new LocationSampler(r_update, nRadial, nAngular);
    LocationSampler *samplerForSearch =
            new LocationSampler(r_search, nRadial_search, nAngular_search);

    ObjDetectorStruck tracker(olarank, features, samplerForSearch,
                              samplerForUpdate,
                   useObjectness, scalePrior, useFilter, pretraining, display);

    tracker.useStraddling = useStraddling;
    tracker.useEdgeDensity = useEdgeDensity;

    int measurementSize = 6;
    arma::colvec x_k(measurementSize, fill::zeros);
    x_k(0) = 0;
    x_k(1) = 0;
    x_k(2) = 0;
    x_k(3) = 0;

    int robustConstant_b = 10;

    int R_cov = 10;
    int Q_cov = 13;
    int P = 13;

    KalmanFilter_my filter =
            KalmanFilterGenerator::generateConstantVelocityFilter(
                    x_k, 0, 0, R_cov, Q_cov, P, robustConstant_b);

    tracker.setFilter(filter);

    tracker.setNote(note_);

    return tracker;
}
