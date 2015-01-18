//
//  Spline.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//

#ifndef __Robust_tracking_by_detection__Spline__
#define __Robust_tracking_by_detection__Spline__

#include <stdio.h>
#include <math.h>       /* pow */
#include <vector>
#include "PiecewiseSpline.h"
#include "armadillo"
    

class Spline{
    

    
    friend std::ostream& operator<<(std::ostream&, const Spline&);

    

    
public:
    std::vector<PiecewiseSpline> s;
    
    double h;
    int n;              // number of points
    double xmin;        // min x value
    double xmax;        // max x value
    
    Spline(){};
    
    void fitSpline(arma::rowvec& x, arma::rowvec& y);
    
    double evaluate(double x);
    };




#endif /* defined(__Robust_tracking_by_detection__Spline__) */
