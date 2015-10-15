//
//  Objectness.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/4/15.
//
//

#ifndef __Robust_tracking_by_detection__Objectness__
#define __Robust_tracking_by_detection__Objectness__

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "armadillo"
#include "SuperPixels.h"

class Straddling{
    int nSuperPixels;
    double inner_threshold;
    std::vector<double> straddling_history;
    
    arma::Cube<int> integrals;
    
public:
    
    cv::Mat canvas;
    int display;
    
    Straddling(){};
    
    Straddling(int n,double inner,int display_=2){this->nSuperPixels=n;
        inner_threshold=inner;
        display=display_;};
    
    void addToHistory(double x){
        this->straddling_history.push_back(x);
    }

    int getNumberOfSuperpixel(){
        return this->integrals.n_slices;
    }
    
    std::vector<double> getStraddlingHistory(){
        return straddling_history;
        
    }
    
    static cv::Rect getInnerRect(cv::Rect& r, double threshold);

    std::pair<std::vector<cv::Rect>, std::vector<double>> nonMaxSuppression(
        std::vector<arma::mat>& s,
        int n,
        std::vector<int>& w,
        std::vector<int>& h);

    arma::mat nonMaxSuppression(const arma::mat& s, int n);

    arma::mat getLabels(cv::Mat&);
    arma::rowvec findStraddling(arma::mat& labels,std::vector<cv::Rect>& rects,
                                int translate_x, int translate_y);
    
    double findStraddlingMeasure(arma::mat& labels, cv::Rect& rectangle);
    
    arma::rowvec findStraddlng_fast(arma::mat& labels,
                                    std::vector<cv::Rect>& rects,
                                    int translate_x,int translate_y);

    void preprocessIntegral(cv::Mat& mat);
    void renormalize(std::vector<arma::mat>& s, int r);
    void straddlingOnCube(int n,
                          int m,
                          int center_x,
                          int center_y,
                          const std::vector<int>& R,
                          const std::vector<int>& w, const std::vector<int>& h,
                          std::vector<arma::mat>& s);
    
    void computeIntegralImages(arma::mat& labels);
    
    double computeStraddling(cv::Rect& rect_big);
    
};


class EdgeDensity{
    
    double threshold_1;
    double threshold_2;
    
    double inner_threshold;
    
    std::vector<double> edge_density_history;
    
    arma::Mat<int> edges_x;
    arma::Mat<int> edges_y;

public:
    cv::Mat canvas;
    int display;
    
    
    void addToHistory(double x){
        this->edge_density_history.push_back(x);
    }
    
    std::vector<double> getEdgeDensityHistory(){
        return edge_density_history;
    }
    
    void setInnerThreshold(double t){
        this->inner_threshold=t;
    }
    
    EdgeDensity(){};
    
    EdgeDensity(double t1,double t2, double inner, int display_){
        this->threshold_1=t1; this->threshold_2=t2;
        this->inner_threshold=inner;
        this->display=display_;};
    
    cv::Mat getEdges(cv::Mat&);
    void computeIntegrals(cv::Mat& labels);
    double computeEdgeDensity(cv::Rect& rect);
    void preprocessIntegral(cv::Mat& image);

    void edgeOnCube(int n,
                    int m,
                    int center_x,
                    int center_y,
                    const std::vector<int>& R,
                    const std::vector<int>& w, const std::vector<int>& h,
                    std::vector<arma::mat>& s);

    arma::rowvec findEdgeObjectness(std::vector<cv::Rect>& rects,
                                    int translate_x, int translate_y);
};

#endif /* defined(__Robust_tracking_by_detection__Objectness__) */
