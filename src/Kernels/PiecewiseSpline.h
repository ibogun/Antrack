//
//  PiecewiseSpline.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//
#include <string>
#ifndef Robust_tracking_by_detection_PiecewiseSpline_h
#define Robust_tracking_by_detection_PiecewiseSpline_h

class PiecewiseSpline {
    
    
    friend std::ostream& operator<<(std::ostream&, const PiecewiseSpline&);
    const double a,b,c,d,x_i;

public:
    

    PiecewiseSpline( double a_, double b_, double c_, double d_,double x_i_): a(a_),b(b_),
    c(c_),d(d_), x_i(x_i_){}
    
    double evaluate(double x){
        return a*(pow(x-x_i, 3))+b*(pow(x-x_i, 2))+c*(x-x_i)+d;
    }
};
#endif
