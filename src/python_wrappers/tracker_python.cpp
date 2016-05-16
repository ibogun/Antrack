#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "armadillo"
#include <vector>

#include "../../src/Tracker/LocationSampler.h"
#include "../../src/Tracker/OLaRank_old.h"
#include <glog/logging.h>

#include "../../src/Filter/KalmanFilterGenerator.h"

#include "../../src/Features/AllFeatures.h"
#include "../../src/Kernels/AllKernels.h"
#include "../../src/Tracker/AllTrackers.h"
#include <boost/python.hpp>

class Antrack {
  public:
    Struck *tracker;

    boost::python::list track(std::string filename) {
        boost::python::list output;
        cv::Rect r = this->tracker->track(filename);

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    boost::python::list calculateDiscriminativeFunction(std::string filename) {

        boost::python::list output;

        cv::Mat image = cv::imread(filename);
        arma::mat discr = this->tracker->calculateDiscriminativeFunction(image);

        for (int i = 0; i < discr.n_rows; i++) {
            boost::python::list out;
            for (int j = 0; j < discr.n_cols; j++) {
                out.append(discr(i, j));
            }
            output.append(out);
        }
        return output;
    }

    void initialize(std::string filename, int x, int y, int width, int height) {

        this->tracker->initialize(filename, x, y, width, height);
    }

    void initializeTracker() {
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

        Feature *features;
        Kernel *kernel;
        std::string featureSTR = "hist";
        std::string kernelSTR = "int";
        features = new HistogramFeatures(4, 16);

        kernel = new IntersectionKernel_fast;

        //    MultiFeature* features=new MultiFeature(fs);

        int verbose = 0;
        int display = 0;
        int m = features->calculateFeatureDimension();

        OLaRank_old *olarank = new OLaRank_old(kernel);
        olarank->setParameters(p, B, m, verbose);

        int r_search = 45;
        int r_update = 60;

        bool useObjectness = false;

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        LocationSampler *samplerForUpdate =
            new LocationSampler(r_update, nRadial, nAngular);
        LocationSampler *samplerForSearch =
            new LocationSampler(r_search, nRadial_search, nAngular_search);

        this->tracker =
            new Struck(olarank, features, samplerForSearch, samplerForUpdate,
                       false, false, true, false, false);

        int measurementSize = 6;
        arma::colvec x_k(measurementSize, fill::zeros);
        x_k(0) = 0;
        x_k(1) = 0;
        x_k(2) = 0;
        x_k(3) = 0;

        int robustConstant_b = 10;

        int R_cov = 13;
        int Q_cov = 13;
        int P = 10;

        KalmanFilter_my filter =
            KalmanFilterGenerator::generateConstantVelocityFilter(
                x_k, 0, 0, R_cov, Q_cov, P, robustConstant_b);

        this->tracker->setFilter(filter);

        this->tracker->setNote("");
    }

    void initializeTrackerWithParameters(std::string featureSTR,
                                         std::string kernelSTR, int C, int n_O,
                                         int n_R, int nRadial, int nAngular,
                                         int B, int nRadial_search,
                                         int nAngular_search, int L, int bins,
                                         int r_search, int r_update,
                                         int display) {
        // Parameters
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        int robustConstant_b = 10;
        int R_cov = 13;
        int Q_cov = 13;
        int P = 10;

        params p;
        p.C = C;
        p.n_O = n_O;
        p.n_R = n_R;
        bool useObjectness = false;
        Feature *features;
        Kernel *kernel;

        features = new HistogramFeatures(L, bins);

        kernel = new IntersectionKernel_fast;

        int verbose = 0;

        int m = features->calculateFeatureDimension();

        OLaRank_old *olarank = new OLaRank_old(kernel);
        olarank->setParameters(p, B, m, verbose);

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        LocationSampler *samplerForUpdate =
            new LocationSampler(r_update, nRadial, nAngular);
        LocationSampler *samplerForSearch =
            new LocationSampler(r_search, nRadial_search, nAngular_search);

        this->tracker =
            new Struck(olarank, features, samplerForSearch, samplerForUpdate,
                       false, false, true, false, false);

        int measurementSize = 6;
        arma::colvec x_k(measurementSize, fill::zeros);
        x_k(0) = 0;
        x_k(1) = 0;
        x_k(2) = 0;
        x_k(3) = 0;

        KalmanFilter_my filter =
            KalmanFilterGenerator::generateConstantVelocityFilter(
                x_k, 0, 0, R_cov, Q_cov, P, robustConstant_b);

        this->tracker->setFilter(filter);

        this->tracker->setNote("");
    }

    ~Antrack() { delete tracker; }
};

class RobStruck {
  public:
    Struck *tracker;
    std::unordered_map<std::string, std::string> featureParamsMap;

    boost::python::list track(std::string filename) {
        boost::python::list output;
        cv::Rect r = this->tracker->track(filename);

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    void initialize(std::string filename, int x, int y, int width, int height) {

        this->tracker->initialize(filename, x, y, width, height);
    }

    ~RobStruck() { delete tracker; }

    void deepFeatureParams(std::string folder) {

        std::string proto = folder + "/imagenet_memory.prototxt";
        std::string weights = folder + "/bvlc_reference_caffenet.caffemodel";

        this->featureParamsMap.insert(std::make_pair("proto", proto));
        this->featureParamsMap.insert(std::make_pair("weights", weights));
    }

    void createTracker(std::string kernel, std::string feature, int filter) {

        bool pretraining = false;
        bool useEdgeDensity = false;
        bool useStraddling = false;
        bool scalePrior = false;

        std::string note = "RobStruck tracker";

        bool useFilter;

        if (filter > 0) {
            useFilter = true;
        } else {
            useFilter = false;
        }

        this->tracker =
            new Struck(pretraining, useFilter, useEdgeDensity, useStraddling,
                       scalePrior, kernel, feature, note);

        this->tracker->setFeatureParams(featureParamsMap);
    }

    void setLocationSamplerParameters(int nRadial, int nAngular) {
        this->tracker->setLocationSamplerParameters(nRadial, nAngular);
    }

    void setDisplay(int display) {
        CHECK_NOTNULL(tracker);
        this->tracker->display = display;
    }

    void killDisplay() { cv::destroyAllWindows(); }
};

class ObjStruck {
  public:
    ObjDetectorStruck *tracker;

    boost::python::list track(std::string filename) {
        boost::python::list output;
        cv::Rect r = this->tracker->track(filename);

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    void initialize(std::string filename, int x, int y, int width, int height) {
        this->tracker->initialize(filename, x, y, width, height);
    }

    boost::python::list applyDetectorFunctionOnMatrix(std::string filename,
                                                      int x, int y, int width,
                                                      int height, int delta) {

        LOG(INFO) <<" image: "<< filename;

        cv::Mat image = cv::imread(filename);

        int x_min = max(0, x - delta);
        int y_min = max(0, y - delta);
        int x_max = min(image.cols, x + width + delta);
        int y_max = min(image.rows, y + height + delta);
        cv::Rect rect(x, y, width, height);

        cv::Rect big_box(x_min, y_min, x_max - x_min, y_max - y_min);
        LOG(INFO) << big_box;
        cv::Mat small_image(image, big_box);
        LOG(INFO) << "Starting discr business";
        arma::mat discr =
            this->tracker->applyDetectorFunctionOnMatrix(small_image, rect);

        boost::python::list output;

        for (int i = 0; i < discr.n_rows; i++) {
            boost::python::list row;
            for (int j = 0; j < discr.n_cols; j++) {
                row.append(discr(i, j));
            }
            output.append(row);
        }

        return output;
    }

    ~ObjStruck() { delete tracker; }

    void createTracker(std::string kernel, std::string feature, int filter,
                       double lambda_e, double lambda_s, double inner,
                       double straddling_threshold) {
        bool pretraining = false;
        bool useEdgeDensity = false;
        bool useStraddling = false;
        bool scalePrior = false;

        std::string note = "ObjStruck tracker";

        bool useFilter;

        if (filter > 0) {
            useFilter = true;
        } else {
            useFilter = false;
        }

        this->tracker = new ObjDetectorStruck(
            pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
            kernel, feature, note);

        std::unordered_map<std::string, double> map;

        CHECK_GE(lambda_s, 0);
        CHECK_GE(lambda_e, 0);
        CHECK_LE(lambda_s, 1);
        CHECK_LE(lambda_e, 1);

        CHECK_GT(inner, 0);
        CHECK_LE(inner, 1);
        CHECK_GE(straddling_threshold, 0);

        map.insert(std::make_pair("lambda_straddling", lambda_s));
        map.insert(std::make_pair("lambda_edgeness", lambda_e));
        map.insert(std::make_pair("inner", inner));
        map.insert(
            std::make_pair("straddling_threshold", straddling_threshold));

        this->tracker->setParams(map);
    }

    void setDisplay(int display) {
        CHECK_NOTNULL(tracker);
        this->tracker->display = display;
    }

    void killDisplay() { cv::destroyAllWindows(); }
};

class MStruck {
  public:
    MBestStruck *tracker;
    std::unordered_map<std::string, std::string> featureParamsMap;

    boost::python::list track(std::string filename) {
        boost::python::list output;

        cv::Mat image = cv::imread(filename);
        cv::Rect r = this->tracker->track(image);

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    void initialize(std::string filename, int x, int y, int width, int height) {
        cv::Rect r(x, y, width, height);
        cv::Mat image = cv::imread(filename);
        this->tracker->initialize(image, r);
    }

    ~MStruck() { delete tracker; }

    void createTracker(std::string kernel, std::string feature, int filter,
                       std::string dis_features, std::string dis_kernel,
                       std::string top_features, std::string top_kernel) {
        bool pretraining = false;
        bool useEdgeDensity = true;
        bool useStraddling = true;
        bool scalePrior = false;

        std::string note = "MBest tracker";

        bool useFilter = true;

        this->tracker =
            new MBestStruck(pretraining, useFilter, useEdgeDensity,
                            useStraddling, scalePrior, kernel, feature, note);

        this->tracker->setUpdateNFrames(3);

        featureParamsMap.insert(std::make_pair("dis_features", dis_features));
        featureParamsMap.insert(std::make_pair("dis_kernel", dis_kernel));
        featureParamsMap.insert(std::make_pair("top_features", top_features));
        featureParamsMap.insert(std::make_pair("top_kernel", top_kernel));
        this->tracker->setFeatureParams(featureParamsMap);
    }

    void deepFeatureParams(std::string folder) {

        std::string proto = folder + "/imagenet_memory.prototxt";
        std::string weights = folder + "/bvlc_reference_caffenet.caffemodel";

        this->featureParamsMap.insert(std::make_pair("proto", proto));
        this->featureParamsMap.insert(std::make_pair("weights", weights));
    }

    void setObjectnessParams(double staddling, double edge) {
        this->tracker->ObjDetectorStruck::setLambda(staddling, edge);
    }

    void setDisplay(int display) {
        CHECK_NOTNULL(tracker);
        this->tracker->display = display;
    }

    void killDisplay() { cv::destroyAllWindows(); }
    void setM(int M_) { this->tracker->setM(M_); }
    void setLambda(double lambda_) { this->tracker->setLambda(lambda_); }
    void setTopBudget(int B_) { this->tracker->setTopBudget(B_); }
    void setBottomBudget(int B_) { this->tracker->setBottomBudget(B_); }
};

class ScaleDeepStruck {
  public:
    ScaleStruck *tracker;
    std::unordered_map<std::string, std::string> featureParamsMap;

    boost::python::list track(std::string filename) {
        boost::python::list output;

        cv::Mat image = cv::imread(filename);
        cv::Rect r = this->tracker->track(image);

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    void initialize(std::string filename, int x, int y, int width, int height) {
        cv::Rect r(x, y, width, height);
        cv::Mat image = cv::imread(filename);
        this->tracker->initialize(image, r);
    }

    ~ScaleDeepStruck() { delete tracker; }

    void createTracker(std::string kernel, std::string feature, int filter,
                       std::string dis_features, std::string dis_kernel,
                       std::string top_features, std::string top_kernel) {
        bool pretraining = false;
        bool useEdgeDensity = true;
        bool useStraddling = true;
        bool scalePrior = false;

        std::string note = "Scale Deep Struck tracker";

        bool useFilter = true;

        this->tracker =
            new ScaleStruck(pretraining, useFilter, useEdgeDensity,
                            useStraddling, scalePrior, kernel, feature, note);

        this->tracker->setUpdateNFrames(3);

        featureParamsMap.insert(std::make_pair("dis_features", dis_features));
        featureParamsMap.insert(std::make_pair("dis_kernel", dis_kernel));
        featureParamsMap.insert(std::make_pair("top_features", top_features));
        featureParamsMap.insert(std::make_pair("top_kernel", top_kernel));
        this->tracker->setFeatureParams(featureParamsMap);
    }

    void deepFeatureParams(std::string folder) {

        std::string proto = folder + "/imagenet_memory.prototxt";
        std::string weights = folder + "/bvlc_reference_caffenet.caffemodel";

        this->featureParamsMap.insert(std::make_pair("proto", proto));
        this->featureParamsMap.insert(std::make_pair("weights", weights));
    }

    void setObjectnessParams(double staddling, double edge) {
        this->tracker->ObjDetectorStruck::setLambda(staddling, edge);
    }

    void setDisplay(int display) {
        CHECK_NOTNULL(tracker);
        this->tracker->display = display;
    }

    void killDisplay() { cv::destroyAllWindows(); }
    void setM(int M_) { this->tracker->setM(M_); }
    void setLambda(double lambda_) { this->tracker->setLambda(lambda_); }
    void setTopBudget(int B_) { this->tracker->setTopBudget(B_); }
    void setBottomBudget(int B_) { this->tracker->setBottomBudget(B_); }
};

using namespace boost::python;

BOOST_PYTHON_MODULE(antrack)

{
    class_<Antrack>("Antrack")
        .def("initialize", &Antrack::initialize)
        .def("track", &Antrack::track)
        .def("initializeTrackerWithParameters",
             &Antrack::initializeTrackerWithParameters)
        .def("calculateDiscriminativeFunction",
             &Antrack::calculateDiscriminativeFunction)
        .def("initializeTracker", &Antrack::initializeTracker);

    class_<RobStruck>("RobStruck")
        .def("initialize", &RobStruck::initialize)
        .def("createTracker", &RobStruck::createTracker)
        .def("track", &RobStruck::track)
        .def("setDisplay", &RobStruck::setDisplay)
        .def("setLocationSamplerParameters",
             &RobStruck::setLocationSamplerParameters)
        .def("deepFeatureParams", &RobStruck::deepFeatureParams)
        .def("killDisplay", &RobStruck::killDisplay);

    class_<ObjStruck>("ObjStruck")
        .def("initialize", &ObjStruck::initialize)
        .def("createTracker", &ObjStruck::createTracker)
        .def("track", &ObjStruck::track)
        .def("applyDetectorFunctionOnMatrix",
             &ObjStruck::applyDetectorFunctionOnMatrix)
        .def("setDisplay", &ObjStruck::setDisplay)
        .def("killDisplay", &ObjStruck::killDisplay);

    class_<MStruck>("MStruck")
        .def("initialize", &MStruck::initialize)
        .def("deepFeatureParams", &MStruck::deepFeatureParams)
        .def("createTracker", &MStruck::createTracker)
        .def("track", &MStruck::track)
        .def("setM", &MStruck::setM)
        .def("setLambda", &MStruck::setLambda)
        .def("setTopBudget", &MStruck::setTopBudget)
        .def("setBottomBudget", &MStruck::setBottomBudget)
        .def("setDisplay", &MStruck::setDisplay)
        .def("setObjectnessParams", &MStruck::setObjectnessParams)
        .def("killDisplay", &MStruck::killDisplay);

    class_<ScaleDeepStruck>("ScaleDeepStruck")
        .def("initialize", &ScaleDeepStruck::initialize)
        .def("deepFeatureParams", &ScaleDeepStruck::deepFeatureParams)
        .def("createTracker", &ScaleDeepStruck::createTracker)
        .def("track", &ScaleDeepStruck::track)
        .def("setTopBudget", &ScaleDeepStruck::setTopBudget)
        .def("setBottomBudget", &ScaleDeepStruck::setBottomBudget)
        .def("setDisplay", &ScaleDeepStruck::setDisplay)
        .def("setObjectnessParams", &ScaleDeepStruck::setObjectnessParams);
}
// find how to write functions which return some values in c++/python boost
// framework
