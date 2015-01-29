//
//  KalmanFilter.cpp
//  Structured_BING
//
//  Created by Ivan Bogun on 8/28/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "KalmanFilter.h"


/**
 *  Initialize Kalman filter structure
 */
KalmanFilter_my::KalmanFilter_my(arma::mat& F_,arma::mat& H_,arma::mat& Q_,arma::mat& R_,
                           arma::mat& P_0,arma::colvec& x_0,int im_w,int im_h,double b_){
    
    this->F=F_;
    this->H=H_;
    this->Q=Q_;
    this->R=R_;
    this->P_kk=P_0;
    this->x_kk=x_0;
    this->b=b_;
    
    this->im_height=im_h;
    this->im_width=im_w;
    
    this->n=x_0.n_rows;
    
}


cv::Rect KalmanFilter_my::getBoundingBox(int bb_width, int bb_height,arma::colvec x){
    
    arma::colvec x_k=x;
    

    int n=this->im_width;
    int m=this->im_height;

    
    for (int jj=0; jj<x_k.size(); jj++) {
        x_k(jj)=cvRound(x_k(jj));
    }
    
    x_k(0)=MAX(0,x_k(0));
    x_k(1)=MAX(0,x_k(1));
    
    if (x_k(2)<bb_width) {
        x_k(2)=bb_width;
    }
    
    if (x_k(3)<bb_height) {
        x_k(3)=bb_height;
    }
    
    if(x_k(2)>n-1){
        x_k(0)=n-1-(x_k(2)-x_k(0));
        x_k(2)=n-1;
    }
    
    if(x_k(3)>m-1){
        x_k(1)=m-1-(x_k(3)-x_k(1));
        x_k(3)=m-1;
    }
    
    cv::Rect bestLocationFilter(x_k(0),x_k(1),x_k(2)-x_k(0),x_k(3)-x_k(1));
    return bestLocationFilter;
}

/**
 *  Given measurement predict value of the values WITHOUT updating the filter(without correction step)
 *
 *  @param z_k measurements matrix
 *
 *  @return prediction of the variables
 */
arma::colvec KalmanFilter_my::predict(arma::colvec& z_k){
    
    using namespace arma;
    
    // prediction step
    colvec x_k_hat_a=this->F*this->x_kk;
    mat P_k_a=this->F*this->P_kk*(this->F).t()+this->Q;
    
    // correction step
    
    mat K_k=P_k_a*((this->H).t())*inv(this->H*P_k_a*((this->H).t())+this->R);
    
    mat correction=K_k*(z_k- this->H*x_k_hat_a);
    double robustConstant=std::min(1.0, this->b/(norm(correction)));
    
    colvec x_k=x_k_hat_a+correction*robustConstant;
    
    
    return x_k;
}


arma::colvec KalmanFilter_my::predictAndCorrect(arma::colvec& z_k){
    using namespace arma;
    
    // prediction step
    colvec x_k_hat_a=this->F*this->x_kk;
    mat P_k_a=this->F*this->P_kk*(this->F).t()+this->Q;
    
    // correction step
    
    mat K_k=P_k_a*((this->H).t())*inv(this->H*P_k_a*((this->H).t())+this->R);
    
    mat correction=K_k*(z_k- this->H*x_k_hat_a);
    double robustConstant=std::min(1.0, b/(norm(correction)));
    
    colvec x_k=x_k_hat_a+correction*robustConstant;
    
    mat P_k=(eye(this->n, this->n)-K_k*this->H)*P_k_a;
    
    
    this->P_kk=P_k;
    this->K_kk=K_k;
    this->x_kk=x_k;
    
    return x_k;
}



std::ostream& operator<<(std::ostream &strm, const KalmanFilter_my &f){
    std::string line="--------------------------------------------------------\n";
    strm<<"F: \n"<<f.F<<line;
    strm<<"H: \n"<<f.H<<line;
    strm<<"Q: \n"<<f.Q<<line;
    strm<<"R: \n"<<f.R<<line;
    strm<<"input vector dim  : "<<f.n<<"\n";
    strm<<"robust constant, b: "<<f.b<<"\n";
    
    return strm;
}











