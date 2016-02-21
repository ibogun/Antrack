#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "armadillo"
#include <vector>


#include <glog/logging.h>
#include "../../src/Tracker/OLaRank_old.h"
#include "../../src/Tracker/LocationSampler.h"

#include "../../src/Filter/KalmanFilterGenerator.h"


#include "../../src/Tracker/AllTrackers.h"
#include "../../src/Kernels/AllKernels.h"
#include "../../src/Features/AllFeatures.h"
#include <boost/python.hpp>

class Antrack {
 public:
    Struck* tracker;


  boost::python::list track(std::string filename) {
      boost::python::list output;
      cv::Rect r = this->tracker->track(filename);

      output.append(r.x);
      output.append(r.y);
      output.append(r.width);
      output.append(r.height);
        return output;
  }

  boost::python::list calculateDiscriminativeFunction(std::string filename){

    boost::python::list output;

    cv::Mat image = cv::imread(filename);
    arma::mat discr = this->tracker->calculateDiscriminativeFunction(image);

    for (int i = 0; i < discr.n_rows; i++) {
      boost::python::list out;
      for (int j = 0; j < discr.n_cols; j++) {
        out.append(discr(i,j));
      }
      output.append(out);
    }
    return output;
  }

  void initialize(std::string filename, int x, int y,
                  int width, int height) {

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

      this->tracker = new Struck(olarank, features, samplerForSearch,
                                 samplerForUpdate,
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
                                       std::string kernelSTR,
                                       int C,
                                       int n_O,
                                       int n_R,
                                       int nRadial,
                                       int nAngular,
                                       int B,
                                       int nRadial_search,
                                       int nAngular_search,
                                       int L,
                                       int bins,
                                       int r_search,
                                       int r_update,
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

      this->tracker = new Struck(olarank, features, samplerForSearch,
                                 samplerForUpdate,
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

~Antrack(){
    delete tracker;
}

};


class RobStruck {
 public:
  Struck* tracker;


  boost::python::list track(std::string filename) {
    boost::python::list output;
    cv::Rect r = this->tracker->track(filename);

    output.append(r.x);
    output.append(r.y);
    output.append(r.width);
    output.append(r.height);
    return output;
  }

  void initialize(std::string filename, int x, int y,
                  int width, int height) {
    this->tracker->initialize(filename, x, y, width, height);
  }


  ~RobStruck() {
    delete tracker;
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

    this->tracker = new Struck(pretraining, useFilter, useEdgeDensity,
                               useStraddling, scalePrior, kernel, feature,
                               note);
  }

  void setDisplay(int display) {
    CHECK_NOTNULL(tracker);
    this->tracker->display = display;
  }

  void killDisplay() {
    cv::destroyAllWindows();
  }
};

class ObjStruck {
 public:
  ObjDetectorStruck* tracker;


  boost::python::list track(std::string filename) {
    boost::python::list output;
    cv::Rect r = this->tracker->track(filename);

    output.append(r.x);
    output.append(r.y);
    output.append(r.width);
    output.append(r.height);
    return output;
  }

  void initialize(std::string filename, int x, int y,
                  int width, int height) {
    this->tracker->initialize(filename, x, y, width, height);
  }


  ~ObjStruck() {
    delete tracker;
  }

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

    this->tracker = new ObjDetectorStruck(pretraining, useFilter,
                                          useEdgeDensity,
                                          useStraddling, scalePrior,
                                          kernel, feature,
                                          note);

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

  void killDisplay() {
    cv::destroyAllWindows();
  }
};

using namespace boost::python;

BOOST_PYTHON_MODULE(antrack)

{
  class_<Antrack>("Antrack")
    .def("initialize",&Antrack::initialize)
    .def("track", &Antrack::track)
    .def("initializeTrackerWithParameters", &Antrack::initializeTrackerWithParameters)
    .def("calculateDiscriminativeFunction", &Antrack::calculateDiscriminativeFunction)
    .def("initializeTracker",&Antrack::initializeTracker)
      ;

  class_<RobStruck>("RobStruck")
    .def("initialize", &RobStruck::initialize)
    .def("createTracker", &RobStruck::createTracker)
    .def("track", &RobStruck::track)
    .def("setDisplay", &RobStruck::setDisplay)
    .def("killDisplay", &RobStruck::killDisplay)
    ;

  class_<ObjStruck>("ObjStruck")
    .def("initialize", &ObjStruck::initialize)
    .def("createTracker", &ObjStruck::createTracker)
    .def("track", &ObjStruck::track)
    .def("setDisplay", &ObjStruck::setDisplay)
    .def("killDisplay", &ObjStruck::killDisplay)
    ;
}
// find how to write functions which return some values in c++/python boost
// framework
