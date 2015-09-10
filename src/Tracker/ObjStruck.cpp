
#include "ObjStruck.h"
#include  <glog/logging.h>

cv::Rect ObjectStruck::track(cv::Mat& image){

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

            this->objectnessCanvas = cv::Mat(h, w, image.type(), CV_RGB(0, 0, 0));
    }

    // calculate integral images
    //if (useStraddling) {
    this->straddle.preprocessIntegral(small_image);
    //}

    std::vector<int> radiuses;
    std::vector<int> heights;
    std::vector<int> widths;

    std::vector<arma::mat> straddling_cube =
            this->samplerForSearch->generateBoxesTensor(lastLocation, &radiuses,
                                                        &widths, &heights);


    this->straddle.straddlingOnCube(small_image.rows, small_image.cols,
                                     radiuses, widths, heights,
                                     straddling_cube);

    int non_max_suppression_n = 3;
    std::pair<std::vector<cv::Rect>, std::vector<double>> suppressed =
        this->straddle.nonMaxSuppression(straddling_cube, non_max_suppression_n,
                                         widths, heights);

    cv::Rect image_box(0,0, image.cols, image.rows);
    for (int i =0 ; i< suppressed.first.size(); i++) {
        cv::Rect rect(suppressed.first[i].x + x_min,
                      suppressed.first[i].y + y_min,
                      suppressed.first[i].width, suppressed.first[i].height);

        cv::Point top_left(rect.x, rect.y);
        cv::Point bottom_right(rect.x + rect.width, rect.y + rect.height);
        if(image_box.contains(top_left) && image_box.contains(bottom_right)){
             locationsOnaGrid.push_back(rect);
        }
    }

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
                                                   this->lastLocation.height, x_k);

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

    if (updateTracker && this->boundingBoxes.size() % this->updateEveryNframes==0) {

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
    }
    /**
     End of tracker udpate
     **/

    if (display == 1) {
        cv::Scalar color(255, 0, 0);
        cv::Mat plotImg = image.clone();
        cv::rectangle(plotImg, lastLocation, color, 2);

        if (useFilter) {
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255, 100), 0);
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
            cv::rectangle(plotImg, bestLocationFilter, cv::Scalar(0, 255, 100), 0);
        }


        if (this->useEdgeDensity || this->useStraddling) {

            cv::rectangle(plotImg, lastLocationObjectness, cv::Scalar(0, 204, 102),
                          0);

            cv::Mat objPlot = this->objPlot->getCanvas();
            cv::resize(objPlot, objPlot, cv::Size(image.cols, image.rows));

            cv::Rect bottomLeftRect(image.rows + 1, 0, objPlot.rows, objPlot.cols);
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

ObjectStruck ObjectStruck::getTracker(bool pretraining, bool useFilter, bool useEdgeDensity,
                                      bool useStraddling, bool scalePrior,
                                      std::string kernelSTR, std::string featureSTR, std::string note_) {

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
        std::cout<<f1->calculateFeatureDimension()<<std::endl;

        cv::Size winSize(32, 32);
        cv::Size blockSize(16, 16);
        cv::Size cellSize(4, 4); // was 8
        cv::Size blockStride(16, 16);
        int nBins = 8;          // was 5

        f2= new HoG(winSize,blockSize,cellSize,blockSize,nBins);

        std::cout<<f2->calculateFeatureDimension()<<std::endl;


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

    // IntersectionKernel_fast* kernel=new IntersectionKernel_fast;
    // ApproximateKernel* kernel= new ApproximateKernel(30);
    // IntersectionKernel* kernel=new IntersectionKernel;

    // RBFKernel *kernel = new RBFKernel(0.2);

    // HoGandRawFeatures* features=new HoGandRawFeatures(size,16);
    // LinearKernel* kernel=new LinearKernel;

    //    Haar* f1=new Haar(2);
    //    HistogramFeatures* f2=new HistogramFeatures(2,16);
    //    IntersectionKernel* k2=new IntersectionKernel;
    //    RBFKernel* k1=new RBFKernel(0.2);
    //
    //    std::vector<Kernel*> ks;
    //
    //    ks.push_back(k1);
    //    ks.push_back(k2);
    //
    //    std::vector<Feature*>    fs;
    //
    //    fs.push_back(f1);
    //    fs.push_back(f2);
    //
    //    MultiKernel* kernel=new MultiKernel(ks,fs);
    //    MultiFeature* features=new MultiFeature(fs);

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

    ObjectStruck tracker(olarank, features, samplerForSearch, samplerForUpdate,
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

    int R_cov = 5;
    int Q_cov = 5;
    int P = 3;

    KalmanFilter_my filter =
            KalmanFilterGenerator::generateConstantVelocityFilter(
                    x_k, 0, 0, R_cov, Q_cov, P, robustConstant_b);

    tracker.setFilter(filter);

    tracker.setNote(note_);

    return tracker;
}
