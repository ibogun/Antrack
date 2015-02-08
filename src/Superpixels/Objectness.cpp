//
//  Objectness.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/4/15.
//
//

#include "Objectness.h"


arma::mat Straddling::getLabels(cv::Mat& image){
    
    SuperPixels seeds;
    
    arma::mat Label=seeds.calculateSegmentation_armamat(image, this->nSuperPixels);
    
    return Label;
}


double Straddling::findStraddlingMeasure(arma::mat &labels, cv::Rect &rect){
    
    arma::mat box=labels.submat(rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);
    
    arma::mat uniqueLabels=arma::unique(box);
    
    //    for (int i=0; i<uniqueLabels.n_elem; i++) {
    //        std::cout<<uniqueLabels(i)<<std::endl;
    //    }
    
    using namespace arma;
    
    
    double measure=0;
    
    // for every label do
    // get label
    //int label=uniqueLabels(0);
    //std::cout<<"Label: "<<label<<std::endl;
    for (int i=0; i<uniqueLabels.size(); i++) {
        int label=uniqueLabels(i);
        
        // get all pixels for that label
        uvec pixelsForLabel=arma::find(labels==label);
        
        // get all pixels for that label in the box
        uvec pixelsForLabelInBox=arma::find(box==label);
        
        //cout<<pixelsForLabel.size()<<" "<<pixelsForLabelInBox.size()<<endl;
        
        measure+=MIN(pixelsForLabelInBox.size(), (pixelsForLabel.size()-pixelsForLabelInBox.size()))/((double)(rect.width*rect.height));
        
        //cout<<measure<<endl;
    }
    double straddling=1-measure;
    
    return straddling;
}

arma::rowvec Straddling::findStraddling(arma::mat &labels, std::vector<cv::Rect> &rects,
                                        int translate_x, int translate_y){
    
    
    arma::rowvec measures(rects.size(),arma::fill::zeros);
    
    for (int i=0; i<rects.size(); i++) {
        
        cv::Rect translatedRect(rects[i].x-translate_x,rects[i].y-translate_y,
                                rects[i].width,rects[i].height);
        double m=findStraddlingMeasure(labels, translatedRect);
        measures[i]=m;
    }
    
    return measures;
}


arma::rowvec Straddling::findStraddlng_fast(arma::mat &labels, std::vector<cv::Rect> &rects, int translate_x, int translate_y){
    
    // get unique labels
    arma::mat uniqueLabels=arma::unique(labels);
    
    // allocate matrices for each superpixel
    // assume labels are labelled from 0 to max(labels)
    
    int m=labels.n_cols;
    int n=labels.n_rows;
    
    std::vector<arma::mat> integrals;
    
    for (int i=0; i<uniqueLabels.size(); i++) {
        arma::mat integral(n+1,m+1,arma::fill::zeros);
        int label=uniqueLabels(i);
        
        // calculate integral image here
        for (int j=1; j<m+1; j++) {
            for (int s=1; s<n+1; s++) {
                if (labels(s-1,j-1)==label) {
                    integral(s,j)++;
                }
                
                if (j!=0) {
                    integral(s,j)+=integral(s,j-1);
                }
                
                if (s!=0) {
                    integral(s,j)+=integral(s-1,j);
                    
                    if (j!=0) {
                        integral(s,j)-=integral(s-1,j-1);
                    }
                }

            }
        }
        
        integrals.push_back(integral);
    }
    
    // resulting scores
    arma::rowvec measures(rects.size(),arma::fill::zeros);
    
    for (int i=0; i<rects.size(); i++) {
        
        double measure=0;
        // for each superpixel
        
        for (int superpixel=0; superpixel<uniqueLabels.size(); superpixel++) {
            
            // find area of the overlap between superpixel and window
            cv::Rect rect(rects[i].x-translate_x,rects[i].y-
                          translate_y,rects[i].width,rects[i].height);
            
            int area_superpixel_window_overlap=integrals[superpixel](rect.x+rect.width+1,
                                                                     rect.y+rect.height+1)+
                                            integrals[superpixel](rect.x,rect.y)-
                                            integrals[superpixel](rect.x+rect.width+1,rect.y)-
                                            integrals[superpixel](rect.x,rect.y+rect.height+1);
            
            int area_superpixel_without_window=integrals[superpixel](n,m)-area_superpixel_window_overlap;
            
            measure+=MIN(area_superpixel_window_overlap,
                         area_superpixel_without_window)/
                        ((double)(rect.width*rect.height));
            
        }
        
        measures(i)=1-measure;
    }
    
    return measures;
    
    
}