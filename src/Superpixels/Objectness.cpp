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
    
    arma::mat Label=seeds.calculateSegmentation(image, this->nSuperPixels,this->display);
    
    if (this->display) {
        this->canvas=seeds.canvas;
    }
    
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
        
        int label=uniqueLabels(i);
        
        // calculate integral image here
        
        for (int s=1; s<n+1; s++) {
            for (int j=1; j<m+1; j++) {
                
                
                
                if (labels(s-1,j-1)==label) {
                    integrals(s,j,i)++;
                }
                
                integrals(s,j,i)+=integrals(s,j-1,i);
                
                integrals(s,j,i)+=integrals(s-1,j,i);
                
                integrals(s,j,i)-=integrals(s-1,j-1,i);
                
                
            }
        }
        
    }
    
    
    // resulting scores
    arma::rowvec measures(rects.size(),arma::fill::zeros);
    
    for (int i=0; i<rects.size(); i++) {
        
        double measure=0;
        // for each superpixel
        // find area of the overlap between superpixel and window
        cv::Rect rect_big(rects[i].x-translate_x,rects[i].y-
                          translate_y,rects[i].width,rects[i].height);
        
        cv::Rect rect=getInnerRect(rect_big,this->inner_threshold);
        
        
        for (int superpixel=0; superpixel<uniqueLabels.size(); superpixel++) {
            
            
            //            int A=integrals(rect.x+rect.width,
            //                            rect.y+rect.height,superpixel);
            //
            //            int B=integrals(rect.x,rect.y,superpixel);
            //
            //            int C=integrals(rect.x+rect.width,rect.y,superpixel);
            //
            //            int D=integrals(rect.x,rect.y+rect.height,superpixel);
            
            int area_superpixel_window_overlap=integrals(rect.x+rect.width,
                                                         rect.y+rect.height,superpixel)+
            integrals(rect.x,rect.y,superpixel)-
            integrals(rect.x+rect.width,rect.y,superpixel)-
            integrals(rect.x,rect.y+rect.height,superpixel);
            
            int area_superpixel_without_window=integrals(n,m,superpixel)-area_superpixel_window_overlap;
            
            
            //int sum=integrals(n,m,superpixel);
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
    
    Mat detected_edges(image.rows,image.cols,CV_8U);
    
    blur(image, detected_edges, Size(3,3));
    
    Canny(detected_edges,detected_edges,this->threshold_1,this->threshold_2);
    
//        cv::imshow("edges", detected_edges);
//    
//        cv::waitKey();
//    
//        cv::destroyAllWindows();
    
    return detected_edges;
}


arma::rowvec EdgeDensity::findEdgeObjectness(cv::Mat &labels, std::vector<cv::Rect> &rects, int translate_x, int translate_y){
    
    
    
    
    // calculate integral images for edges in x and y directions
    
    int m=labels.cols;
    int n=labels.rows;
    
    arma::Mat<int> edges_x(n+1,m+1,arma::fill::zeros);
    arma::Mat<int> edges_y(n+1,m+1,arma::fill::zeros);
    
    
    for (int s=1; s<n+1; s++) {
        for (int j=1; j<m+1; j++) {
            if (labels.at<uchar>(s-1, j-1)>0) {
                edges_x(s,j)++;
            }
            
            
            edges_x(s,j)+=edges_x(s,j-1);
            
            
            if (labels.at<uchar>(s-1, j-1)>0) {
                edges_y(s,j)++;
            }
            
            edges_y(s,j)+=edges_y(s-1,j);
            
            
        }
    }
    
    arma::rowvec measures(rects.size(),arma::fill::zeros);
    for (int i=0; i<rects.size(); i++) {
        
        double measure=0;
        
        
        cv::Rect rect(rects[i].x-translate_x,rects[i].y-
                      translate_y,rects[i].width,rects[i].height);

        
        cv::Rect inner_rect=Straddling::getInnerRect(rect,this->inner_threshold);
        
        // calculate how many edges on the perimeter of the inner rectangle

        
        int edges_in_x=edges_x(inner_rect.y,inner_rect.x+inner_rect.width)+edges_x(inner_rect.y+inner_rect.height,inner_rect.x+inner_rect.width)-edges_x(inner_rect.y,inner_rect.x)-edges_x(inner_rect.y+inner_rect.height,inner_rect.x);
        
        int edges_in_y=edges_y(inner_rect.y+inner_rect.height,inner_rect.x)+edges_y(inner_rect.y+inner_rect.height,inner_rect.x+inner_rect.width)-edges_y(inner_rect.y,inner_rect.x)-edges_y(inner_rect.y,inner_rect.x+inner_rect.width);
        
//        int edges_in_x=edges_x(inner_rect.x+inner_rect.width,inner_rect.y)+edges_x(inner_rect.x+inner_rect.width,inner_rect.y+inner_rect.height)-edges_x(inner_rect.x,inner_rect.y)-edges_x(inner_rect.x,inner_rect.y+inner_rect.height);
//        
//        int edges_in_y=edges_y(inner_rect.x,inner_rect.y+inner_rect.height)+edges_y(inner_rect.x+inner_rect.width,inner_rect.y+inner_rect.height)-edges_y(inner_rect.x,inner_rect.y)-edges_y(inner_rect.x+inner_rect.width,inner_rect.y);
        
        measure=(edges_in_x+edges_in_y)/((double)(2*(inner_rect.width+inner_rect.height)));
        
        measures[i]=measure;
    }
    
    
    
    return measures;
    
}



cv::Rect Straddling::getInnerRect(cv::Rect &rect, double inner_threshold){
    // get inner rectangle
    
    int inner_rect_x=rect.x+rect.width*(1-inner_threshold)/(2.0);
    int inner_rect_y=rect.y+rect.height*(1-inner_threshold)/(2.0);
    int inner_width=rect.width*inner_threshold;
    int inner_height=rect.height*inner_threshold;
    
    cv::Rect inner_rect(inner_rect_x,inner_rect_y,inner_width,inner_height);
    
    return inner_rect;
}






























