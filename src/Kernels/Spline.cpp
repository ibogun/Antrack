//
//  Spline.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//

#include "Spline.h"
#include <string>



void Spline::fitSpline(arma::rowvec &x, arma::rowvec &y){
    
    using namespace arma;
    this->n=x.size();
    this->h=x[1]-x[0];
    
    double xmin=x[0];
    double xmax=x[0];
    
    for (int i=0; i<this->n; i++) {
        if (x[i]>xmax) {
            xmax=x[i];
        }
        
        if (x[i]<xmin) {
            xmin=x[i];
        }
    }
   
    this->xmin=xmin;
    this->xmax=xmax;
    
    
    // fit splines
    
    // 1) create tridiagonal matrix
    
    
    mat A(n-2,n-2,fill::zeros);
    vec b(n-2,fill::zeros);
    
    
    double six_h=6*h;
    double six_over_h_square=6/(h*h);
    
    for (int i=1; i<this->n-1; i++) {
        if (i==1) {

            A(0,1)=1;
        }else if (i==this->n-1-1) {
            A(i-1,i-1-1)=1;

        }else{
            
            A(i-1-1,i-1)=1;
            A(i-1,i-1-1)=1;
        }
        
        A(i-1,i-1)=4;
        
        b[i-1]=(y[i-1]-2*y[i]+y[i+1])*six_over_h_square;
    }
    
    
    vec M=solve(A,b);
    
    // M has n-2 elements, M(-1)=0, M(n-2)=0;
    // now create splines
    
    double a,b_,c,d;
    
    

    a=0;
    b_=0;
    c=(y[1]-y[0])/h -(M[0]/6)*h;
    d=y[0];
    
    PiecewiseSpline p(a, b_, c, d, x[0]);
    
    this->s.push_back(p);
    
    for (int i=1; i<=this->n-2; i++) {
        a=(M[i]-M[i-1])/six_h;
        b_=M[i-1];
        c=(y[i+1]-y[i])/h -((M[i]+2*M[i-1])/6)*h;
        d=y[i];
        PiecewiseSpline p1(a, b_, c, d, x[i]);
        this->s.push_back(p1);
    }
    
    
    // last spline
    
    a=0;
    b_=0;
    c=(y[n-1]-y[n-2])/h+M(n-3)*(h/3);
    d=y[n-1];
    
    PiecewiseSpline p_last(a, b_, c, d, x[n-1]);
    
    this->s.push_back(p_last);
    
    
}

double Spline::evaluate(double x){
    
    if (x<xmin) {
        return this->s[0].evaluate(x);
    }
    
    if (x>xmax) {
        return this->s[this->n-1].evaluate(x);
    }
    
    // find appropriate interval
    int splineIdx=floor((x-xmin)/h);
    
    if (h==0) {
        return 0;
    }
    
    return this->s[splineIdx].evaluate(x);
}


std::ostream& operator<<(std::ostream &strm, const PiecewiseSpline &p_spline) {
    
    std::string s="(x-"+std::to_string(p_spline.x_i)+")";
    strm<<"f(x)="<<p_spline.a<<s<<"^3+"<<p_spline.b<<s<<"^2+"<<p_spline.c<<s<<"+"<<p_spline.d<<"\n";
    
    return strm;
}

std::ostream& operator<<(std::ostream &strm,const  Spline &spline) {
    
    using namespace std;
    
    strm<<"Spline in range: ["<<spline.xmin<<","<<spline.xmax<<"] with"<<
    spline.n<<" data points"<< " step="<<spline.h<<"\n";
    
    double x=spline.xmin;
    for (int i=0; i<spline.s.size(); i++) {
        
        strm<<"["<<to_string(x)<<","<<to_string(x+spline.h)<<"]   "<<spline.s[i];
        x=x+spline.h;
    }
    
    
    return strm;
}


