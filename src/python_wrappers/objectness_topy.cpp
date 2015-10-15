#include "../../src/Tracker/Struck.h"
#include "../../src/Superpixels/Objectness.h"
#include "../../src/Tracker/LocationSampler.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "armadillo"
#include <vector>
#include <boost/python.hpp>
#include "glog/logging.h"
class Objectness {

public:
  cv::Mat image;
  cv::Mat small_image;


  Straddling* straddle;
  EdgeDensity* edgeDensity;

  double downsample=1.03;
  int minScale=-2;
  int maxScale=4;

  int min_x;
  int min_y;

  double getStraddling(int x, int y, int width, int height){
        if (x<0){
          x=0;
        }

        if (y<0){
          y=0;
        }
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
        if (x<0){
            x=0;
        }

        if (y<0){
            y=0;
        }

        if (x + width >= image.cols) {
          width = image.cols - x - 1;
        }

        if (y + height >= image.rows) {
          height = image.rows - y - 1;
        }

        cv::Rect rect(x,y,width,height);

        return this->edgeDensity->computeEdgeDensity(rect);
    };

    double getEdgenessMultiscale(int x, int y,int width, int height){

        double sum=0;
        for (int j = minScale ;j <= maxScale; ++j) {

            int width_s=width*pow(downsample,j);
            int height_s=height*pow(downsample,j);

            int new_x=(int)(x+width/2.0-width_s/2.0);
            int new_y=(int)(y+height/2.0-height_s/2.0);

            sum+=getEdgeness(new_x,new_y,width_s,height_s);
        }

        sum=sum/(maxScale-minScale);
        return sum;
    }

    double getStraddlingMultiscale(int x, int y,int width, int height){

        double sum=0;
        for (int j = minScale ;j <= maxScale; ++j) {

            int width_s=width*pow(downsample,j);
            int height_s=height*pow(downsample,j);

            int new_x=(int)(x+width/2.0-width_s/2.0);
            int new_y=(int)(y+height/2.0-height_s/2.0);

            sum+=getStraddling(new_x,new_y,width_s,height_s);
        }

        sum=sum/(maxScale-minScale);
        return sum;
    }

    void plotObjectness(){
      cv::imshow("Image", this->small_image);
      //cv::waitKey();
      //cv::destroyAllWindows();

      cv::imshow("Objectness", this->straddle->canvas);
      cv::waitKey();
      cv::destroyAllWindows();

    };

  void readImage(std::string imName){
    cv::Mat im = cv::imread(imName);

    // setImageSize(im.cols,im.rows);
    this->image = im.clone();
  };


  void smallImage(int R, int x, int y, int width, int height ){
    int delta = R;

    cv::Rect lastLocation(x, y, width, height);
    int x_min = max(0, lastLocation.x - delta);
    int y_min = max(0, lastLocation.y - delta);

    int x_max = min(image.cols, lastLocation.x + lastLocation.width + delta);
    int y_max = min(image.rows, lastLocation.y + lastLocation.height + delta);

    cv::Rect big_box(x_min, y_min, x_max - x_min, y_max - y_min);
    // extract small image from 'image'
    cv::Mat small_image_crop(this->image, big_box);

    this->min_x = x_min;
    this->min_y = y_min;

    this->small_image = small_image_crop.clone();

  }

  boost::python::list process(const int superpixels,
                              const double inner,
                              const int n,
                              const int R, const int scale_R,
                              const int min_size_half,
                              const int min_scales, const int max_scales,
                              const double downsample,
                              const double shrink_one_side_scale,
                              int x, int y, int width, int height){

    cv::Rect lastLocation(x, y, width, height);
    std::vector<int> radiuses;
    std::vector<int> widths;
    std::vector<int> heights;

    this->straddle=new Straddling(superpixels,inner);
    // add boxes and stuff
    std::vector<arma::mat> straddling_cube = LocationSampler::
      generateBoxesTensor(R, scale_R, min_size_half, min_scales,
                          max_scales, downsample, shrink_one_side_scale,
                          this->small_image.rows,
                          this->small_image.cols,
                          lastLocation, &radiuses, &widths, &heights);

    this->straddle->preprocessIntegral(this->small_image);

    CHECK_NOTNULL(&this->small_image);
    this->straddle->straddlingOnCube(this->small_image.rows,
                                     this->small_image.cols,
                                     x - this->min_x,
                                     y - this->min_y,
                                     radiuses, widths, heights,
                                     straddling_cube);

    if ( n > 0){
      std::vector<arma::mat> suppressed_cube;
      for (int i = 0; i<straddling_cube.size(); i++) {
        arma::mat suppressed =
          this->straddle->nonMaxSuppression(straddling_cube[i], n);

        suppressed_cube.push_back(suppressed);
      }

      std::cout << " Done."<< std::endl;
      straddling_cube = suppressed_cube;

    }

    // convert from vector<arma::mat> to python::list<nd a≈grray>
    boost::python::list output;

    for (int i = 0; i<straddling_cube.size(); i++) {
      int rows = straddling_cube[i].n_rows;
      int cols = straddling_cube[i].n_cols;

      boost::python::list np_array;
      for (int j = 0; j<rows; j++) {
        boost::python::list output_row;
        for (int k = 0; k< cols; k++) {
          output_row.append(straddling_cube[i](j,k));
        }
        np_array.append(output_row);
      }
      output.append(np_array);
    }

    return output;

  }


 boost::python::list processEdge(const int superpixels,
                              const double inner,
                              const int n,
                              const int R, const int scale_R,
                              const int min_size_half,
                              const int min_scales, const int max_scales,
                              const double downsample,
                              const double shrink_one_side_scale,
                              int x, int y, int width, int height){

    cv::Rect lastLocation(x, y, width, height);
    std::vector<int> radiuses;
    std::vector<int> widths;
    std::vector<int> heights;

    EdgeDensity* edge = new EdgeDensity(0.66*100, 1.33*100, inner, 0);
    // add boxes and stuff
    std::vector<arma::mat> straddling_cube = LocationSampler::
      generateBoxesTensor(R, scale_R, min_size_half, min_scales,
                          max_scales, downsample, shrink_one_side_scale,
                          this->small_image.rows,
                          this->small_image.cols,
                          lastLocation, &radiuses, &widths, &heights);

    edge->preprocessIntegral(this->small_image);

    CHECK_NOTNULL(&this->small_image);
    edge->edgeOnCube(this->small_image.rows,
                     this->small_image.cols,
                     x - this->min_x,
                     y - this->min_y,
                     radiuses, widths, heights,
                     straddling_cube);


    // convert from vector<arma::mat> to python::list<nd a≈grray>
    boost::python::list output;

    for (int i = 0; i<straddling_cube.size(); i++) {
      int rows = straddling_cube[i].n_rows;
      int cols = straddling_cube[i].n_cols;

      boost::python::list np_array;
      for (int j = 0; j<rows; j++) {
        boost::python::list output_row;
        for (int k = 0; k< cols; k++) {
          output_row.append(straddling_cube[i](j,k));
        }
        np_array.append(output_row);
      }
      output.append(np_array);
    }

    return output;

  }

  void initializeStraddling(int superpixels, double inner){

        this->straddle=new Straddling(superpixels,inner);

        arma::mat labels=this->straddle->getLabels(this->image );

        this->straddle->computeIntegralImages(labels);
       //this->tracker = &t;
  };


  boost::python::list getEdgenessList(boost::python::list& x,
                                      boost::python::list& y,
                                      boost::python::list& width,
                                      boost::python::list& height){

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

  boost::python::list getStraddlingList(boost::python::list& x,
                                        boost::python::list& y,
                                        boost::python::list& width,
                                        boost::python::list& height){

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


~Objectness(){
    delete straddle;
    delete edgeDensity;
}


  void initializeEdgeDensity(double t1, double t2, double inner){
    this->edgeDensity=new EdgeDensity(t1,t2,inner,0);

    cv::Mat edges=this->edgeDensity->getEdges(this->image);

    this->edgeDensity->computeIntegrals(edges);
  };
};


using namespace boost::python;

BOOST_PYTHON_MODULE(objectness_python)

{
  class_<Objectness>("Objectness")
      //.add_property("getStraddling", &Objectness::getStraddling)
      //.add_property("getEdgeness", &Objectness::getEdgeness)
    .def("readImage",&Objectness::readImage)
    .def("smallImage", &Objectness::smallImage)
    .def("processEdge", &Objectness::processEdge)
    .def("process", &Objectness::process)
    .def("initializeStraddling", &Objectness::initializeStraddling)
    .def("initializeEdgeDensity", &Objectness::initializeEdgeDensity)
    .def("plot", &Objectness::plotObjectness)
    .def("getStraddling", &Objectness::getStraddling)
    .def("getEdgeness", &Objectness::getEdgeness)
    .def("getEdgenessList",&Objectness::getEdgenessList)
    .def("getStraddlingList",&Objectness::getStraddlingList)
    .def("getEdgenessMultiscale",&Objectness::getEdgenessMultiscale)
    .def("getStraddlingMultiscale",&Objectness::getStraddlingMultiscale)
    ;
}
// find how to write functions which return some values in c++/python boost
// framework
