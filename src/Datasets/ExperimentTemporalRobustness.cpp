//
// Created by Ivan Bogun on 4/2/15.
//

#include "ExperimentTemporalRobustness.h"
#include "armadillo"



std::vector<std::pair<cv::Rect,int>> ExperimentTemporalRobustness::generateBoundingBoxes(std::vector<cv::Rect>& rects, int n,
                                                                                         int m) {
    std::vector<std::pair<cv::Rect,int>> boxes;



    int frames=rects.size()-1;
    arma::urowvec t=arma::linspace<arma::urowvec>(0,frames,this->segments);


    for (int j=0;j<this->segments;j++) {

        std::pair<cv::Rect,int> p(rects[t[j]],t[j]);
        boxes.push_back(p);
    }

    return boxes;
}

std::ostream& operator<<(std::ostream &strm, const ExperimentTemporalRobustness &f){
    std::string line="--------------------------------------------------------\n";
    strm<<"Experiment on temporal robustness of the tracker (TRE).\n";
    strm<<"Parameters: \nnumber of segments per video="<<f.segments<<"\n";

    return strm;
}


std::string ExperimentTemporalRobustness::getInfo() {
    std::stringstream ss;

    ss<<*this;
    return ss.str();
}