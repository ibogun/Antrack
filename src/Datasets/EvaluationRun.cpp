//
//  EvaluationRun.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#include "EvaluationRun.h"
#include "cmath"

double centerDistance(cv::Rect& r1,cv::Rect& r2){
    
    double x=std::abs(r1.x+r1.width/(2.0)-(r2.x+r2.width/(2.0)));
    double y=std::abs(r1.y+r1.height/(2.0)-(r2.y+r2.height/(2.0)));
    
    
    return sqrt(pow(x, 2)+pow(y, 2));
}


double overlapJaccard(cv::Rect& r1,cv::Rect& r2){
 
    double overlap=(r1 & r2).area();
    
    double overlapJaccard=overlap/(r1.area()+r2.area()-overlap);
    
    return overlapJaccard;
}


void EvaluationRun::evaluate(std::vector<cv::Rect> &gt, std::vector<cv::Rect> &real){
    
    // calculate precision
    double n=MIN(real.size(), gt.size()) ;
    arma::rowvec precision(n,arma::fill::zeros);
    arma::rowvec success(n,arma::fill::zeros);
    
    
    for (int i=0; i<n; i++) {
        // calculate precision
        precision(i)=centerDistance(gt[i],real[i]);
        success(i) = overlapJaccard(gt[i], real[i]);
    }
    
    // calculate metrics
    
    double maxPrecision=50;
    double maxOverlap= 1;
    
    arma::rowvec precision_metric(quant+1,arma::fill::zeros);
    
    arma::rowvec success_metric(quant+1,arma::fill::zeros);
    
    
    double precisionStep=maxPrecision/quant;
    double overlapStep  =maxOverlap/quant;
    
    arma::rowvec precision_threshold = arma::linspace<arma::rowvec>(0, maxPrecision,quant+1);
    arma::rowvec success_threshold   = arma::linspace<arma::rowvec>(0, maxOverlap,quant+1);
    
    
    for (int i=0; i<n; i++) {
        
        for (int j=0; j<precision_threshold.size(); j++) {
            if (precision(i)<=precision_threshold(j)) {
                precision_metric(j)++;
            }
        }
        
        if (precision[i]<=20) {
            this->precision_20++;
        }
        
        if (success[i]>=0.5) {
            this->overlap_half++;
        }

        
        for (int j=0; j<success_threshold.size(); j++) {
            if (success(i)>=success_threshold(j)) {
                success_metric(j)++;
            }
        }
        
    }
    // normalize
    precision_metric/=n;
    success_metric/=n;
    this->precision_20/=n;
    this->overlap_half/=n;

    // integration using midpoint rule
    
    for (int i=0; i<precision_metric.size()-1; i++) {
        
        this->precision_area+=precisionStep*(precision_metric(i)+precision_metric(i+1))/2.0;
        this->overlap_area+=overlapStep*(success_metric(i)+success_metric(i+1))/2.0;
    }
    //this->precision_area/=50;
    
}


std::ostream& operator<<(std::ostream& strm, const EvaluationRun& run){
    
    
    strm<<"Overlap  : "<<run.overlap_half<<" / "<<run.overlap_area<<"\n";
    strm<<"Precision: "<<run.precision_20<<" / "<<run.precision_area<<"\n";
    
    return strm;
}