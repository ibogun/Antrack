#include "../../src/Tracker/Struck.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "armadillo"
#include <vector>

class Objectness {

  double straddling;
  double edgeness;

  cv::Mat canvas;

public:
  double getStraddling() { return this->straddling; };

  double getEdgeness() { return this->edgeness; };

  void plotObjectness() {
    cv::imshow("Objectness", this->canvas);
    cv::waitKey();
    cv::destroyAllWindows();
  };

  void getObjectness(std::string imName, int x, int y, int width, int height) {

    cv::Mat im = cv::imread(imName);

    if (x + width >= im.cols) {
      width = im.cols - x - 1;
    }

    if (y + height >= im.rows) {
      height = im.rows - y - 1;
    }

    cv::Rect rect(x, y, width, height);
    std::vector<cv::Rect> rects;

    rects.push_back(rect);

    // std::cout << rect << std::endl;
    // std::cout << im.rows << " " << im.cols << std::endl;
    // cv::Mat roi(im, rect);
    //
    // cv::imshow("image", roi);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    Struck t = Struck::getTracker();

    arma::rowvec prediction(1, arma::fill::ones);

    int nSuperpixels = 200;
    std::pair<arma::rowvec, arma::rowvec> measures =
        t.weightWithStraddling(im, prediction, rects, nSuperpixels);

    // cv::imshow("image",t.getObjectnessCanvas());
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    this->canvas = t.getObjectnessCanvas();

    this->straddling = measures.first[0];
    this->edgeness = measures.second[0];
  };
};

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(objectness)

{
  class_<Objectness>("Objectness")
      .add_property("getStraddling", &Objectness::getStraddling)
      .add_property("getEdgeness", &Objectness::getEdgeness)
      .def("getObjectness", &Objectness::getObjectness)
      .def("plot", &Objectness::plotObjectness);
}
