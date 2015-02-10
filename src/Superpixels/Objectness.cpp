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
    
    arma::mat Label=seeds.calculateSegmentation(image, this->nSuperPixels);
    
    return Label;
}


double Straddling::findStraddlingMeasure(arma::mat &labels, cv::Rect &rect){
    
    arma::mat box=labels.submat(rect.x, rect.y, rect.x+rect.width-1, rect.y+rect.height-1);
    
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
    
    
    arma::Cube<int> integrals(n+1,m+1,uniqueLabels.size(),arma::fill::zeros);
    
    
    for (int i=0; i<uniqueLabels.size(); i++) {
        
        int label=uniqueLabels[i];
        
        // calculate integral image here
        for (int j=1; j<m+1; j++) {
            for (int s=1; s<n+1; s++) {
                if (labels[s-1,j-1]==label) {
                    integrals[s,j,i]++;
                }
                
                if (j!=0) {
                    integrals[s,j,i]+=integrals[s,j-1,i];
                }
                
                if (s!=0) {
                    integrals[s,j,i]+=integrals[s-1,j,i];
                    
                    if (j!=0) {
                        integrals[s,j,i]-=integrals[s-1,j-1,i];
                    }
                }
                
            }
        }
        
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
            
            
            int area_superpixel_window_overlap=integrals[rect.x+rect.width,
                                                         rect.y+rect.height,superpixel]+
            integrals[rect.x,rect.y,superpixel]-
            integrals[rect.x+rect.width,rect.y,superpixel]-
            integrals[rect.x,rect.y+rect.height,superpixel];
            
            int area_superpixel_without_window=integrals[n,m,superpixel]-area_superpixel_window_overlap;
            
            measure+=MIN(area_superpixel_window_overlap,
                         area_superpixel_without_window)/
            ((double)(rect.width*rect.height));
            
        }
        
        measures(i)=1-measure;
    }
    
    
    return measures;
    
    
}


cv::Mat EdgeDensity::getEdges(cv::Mat& image){
    using namespace cv;
    
    Mat detected_edges;
    
    blur(image, detected_edges, Size(3,3));
    
    Canny(detected_edges,detected_edges,this->threshold_1,this->threshold_2);
    
    //    cv::imshow("edges", detected_edges);
    //
    //    cv::waitKey();
    //
    //    cv::destroyAllWindows();
    
    return detected_edges;
}


arma::rowvec EdgeDensity::findEdgeObjectness(cv::Mat &labels, std::vector<cv::Rect> &rects, int translate_x, int translate_y){
    
    
    // calculate integral images for edges in x and y directions
    
    int m=labels.cols;
    int n=labels.rows;
    
    arma::Mat<int> edges_x(n+1,m+1,arma::fill::zeros);
    arma::Mat<int> edges_y(n+1,m+1,arma::fill::zeros);
    
    for (int j=1; j<m+1; j++) {
        for (int s=1; s<n+1; s++) {
            
            if (labels.at<uchar>(s, j)>0) {
                edges_x[s,j]++;
            }
            
            
            edges_x[s,j]+=edges_x[s,j-1];
            
            edges_x[s,j]+=edges_x[s-1,j];
            edges_x[s,j]+=edges_x[s-1,j-1];
            
            
            if (labels.at<uchar>(s, j)>0) {
                edges_y[s,j]++;
            }
            
            
            edges_y[s,j]+=edges_y[s,j-1];
            
            edges_y[s,j]+=edges_y[s-1,j];
            edges_y[s,j]+=edges_y[s-1,j-1];
            
            
        }
    }
    
    arma::rowvec measures(rects.size(),arma::fill::zeros);
    for (int i=0; i<rects.size(); i++) {
        
        double measure=0;
        
        
        cv::Rect rect(rects[i].x-translate_x,rects[i].y-
                      translate_y,rects[i].width,rects[i].height);
        
        // get inner rectangle
        
        int inner_rect_x=rect.x+rect.width*(1-this->inner_threshold)/(2.0);
        int inner_rect_y=rect.y+rect.height*(1-this->inner_threshold)/(2.0);
        int inner_width=rect.width*this->inner_threshold;
        int inner_height=rect.height*this->inner_threshold;
        
        cv::Rect inner_rect(inner_rect_x,inner_rect_y,inner_width,inner_height);
        
        // calculate how many edges on the perimeter of the inner rectangle
        
        int edges_in_x=edges_x[inner_rect_x+inner_width,inner_rect_y]+edges_x[inner_rect_x+inner_width,inner_rect_y+inner_height]-edges_x[inner_rect_x,inner_rect_y]-edges_x[inner_rect_x,inner_rect_y+inner_height];
        
        int edges_in_y=edges_y[inner_rect_x,inner_rect_y+inner_height]+edges_y[inner_rect_x+inner_width,inner_rect_y+inner_height]-edges_y[inner_rect_x,inner_rect_y]-edges_y[inner_rect_x+inner_width,inner_rect_y];
        
        measure=(edges_in_x+edges_in_y)/((double)(2*(inner_width+inner_height)));
        
        measures[i]=measure;
    }
    
    
    
    return measures;
    
}
































