
#include "../../src/Tracker/Struck.h"
#include "../../src/Superpixels/Objectness.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "armadillo"
#include <vector>
#include <boost/python.hpp>
class Objectness {

public:
    cv::Mat image;


    Straddling* straddle;
    EdgeDensity *edgeDensity;

    double getStraddling(int x, int y, int width, int height){
        if (x + width >= image.cols) {
          width = image.cols - x - 1;
        }

        if (y + height >= image.rows) {
          height = image.rows - y - 1;
        }

        cv::Rect rect(x,y,width,height);

        return this->straddle->computeStraddling(rect);
    };

    double getEdgeness(int x, int y, int width, int height) {
        if (x + width >= image.cols) {
          width = image.cols - x - 1;
        }

        if (y + height >= image.rows) {
          height = image.rows - y - 1;
        }

        cv::Rect rect(x,y,width,height);

        return this->edgeDensity->computeEdgeDensity(rect);
    };

    void plotObjectness(){
        //   cv::imshow("Objectness", this->tracker.getObjectnessCanvas());
        //   cv::waitKey();
        //   cv::destroyAllWindows();

    };

  void readImage(std::string imName){
      cv::Mat im = cv::imread(imName);

     // setImageSize(im.cols,im.rows);
     this->image = im.clone();
  };

  void initializeStraddling(int superpixels, double inner){

        this->straddle=new Straddling(superpixels,inner);

        arma::mat labels=this->straddle->getLabels(this->image );

        this->straddle->computeIntegralImages(labels);
       //this->tracker = &t;
  };


  boost::python::list getEdgenessList(boost::python::list& x,boost::python::list& y, boost::python::list& width, boost::python::list& height){

      boost::python::list output;
      for (int i = 0; i < len(x); ++i)


        {

            int x_val=boost::python::extract<int>(x[i]);
            int y_val=boost::python::extract<int>(y[i]);
            int w_val=boost::python::extract<int>(width[i]);
            int h_val=boost::python::extract<int>(height[i]);


            double r=this->getEdgeness(x_val,y_val,w_val,h_val);
            output.append(r);
        }

        return output;
  }

  boost::python::list getStraddlingList(boost::python::list& x,boost::python::list& y, boost::python::list& width, boost::python::list& height){

      boost::python::list output;
      for (int i = 0; i < len(x); ++i)


        {

            int x_val=boost::python::extract<int>(x[i]);
            int y_val=boost::python::extract<int>(y[i]);
            int w_val=boost::python::extract<int>(width[i]);
            int h_val=boost::python::extract<int>(height[i]);


            double r=this->getStraddling(x_val,y_val,w_val,h_val);
            output.append(r);
        }

        return output;
  }



  void initializeEdgeDensity(double t1, double t2, double inner){
    this->edgeDensity=new EdgeDensity(t1,t2,inner,0);

    cv::Mat edges=this->edgeDensity->getEdges(this->image);

    this->edgeDensity->computeIntegrals(edges);
  };
};


using namespace boost::python;

BOOST_PYTHON_MODULE(objectness)

{
  class_<Objectness>("Objectness")
      //.add_property("getStraddling", &Objectness::getStraddling)
      //.add_property("getEdgeness", &Objectness::getEdgeness)
      .def("readImage",&Objectness::readImage)
      .def("initializeStraddling", &Objectness::initializeStraddling)
      .def("initializeEdgeDensity", &Objectness::initializeEdgeDensity)
      .def("plot", &Objectness::plotObjectness)
      .def("getStraddling", &Objectness::getStraddling)
      .def("getEdgeness", &Objectness::getEdgeness)
      .def("getEdgenessList",&Objectness::getEdgenessList)
      .def("getStraddlingList",&Objectness::getStraddlingList)

      ;
}
// find how to write functions which return some values in c++/python boost
// framework
