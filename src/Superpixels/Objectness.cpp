//
//  Objectness.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/4/15.
//
//

#include "Objectness.h"
#include "glog/logging.h"

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

void Straddling::computeIntegralImages(arma::mat &labels){
    
    // get unique labels
    arma::mat uniqueLabels=arma::unique(labels);
    
    // allocate matrices for each superpixel
    // assume labels are labelled from 0 to max(labels)
    
    int m=labels.n_cols;
    int n=labels.n_rows;
    
    this->integrals.set_size(n+1, m+1, uniqueLabels.size());
    
    this->integrals.fill(0);

    for (int i=0; i<uniqueLabels.size(); i++) {
        
        int label=uniqueLabels(i);
        
        // calculate integral image here
        
        for (int s=1; s<n+1; s++) {
            for (int j=1; j<m+1; j++) {
                
                
                
                if (labels(s-1,j-1)==label) {
                    this->integrals(s,j,i)++;
                }
                
                this->integrals(s,j,i)+=this->integrals(s,j-1,i);
                
                this->integrals(s,j,i)+=this->integrals(s-1,j,i);
                
                this->integrals(s,j,i)-=this->integrals(s-1,j-1,i);
                
                
            }
        }
        
    }
    
}


arma::rowvec Straddling::findStraddlng_fast(arma::mat &labels,
                                            std::vector<cv::Rect> &rects,
                                            int translate_x, int translate_y){

    arma::rowvec measures(rects.size(),arma::fill::zeros);
    for (int i=0; i<rects.size(); i++) {

        cv::Rect rect_big(rects[i].x-translate_x,rects[i].y-
                          translate_y,rects[i].width,rects[i].height);

        measures(i)=computeStraddling(rect_big);
    }
    return measures;
}


void Straddling::preprocessIntegral(cv::Mat& image){

    arma::mat labels = this->getLabels(image);
    this->computeIntegralImages(labels);
}


void Straddling::straddlingOnCube(int img_rows,
                                  int img_cols,
                                  int center_x,
                                  int center_y,
                                  const std::vector<int> &R,
                                  const std::vector<int> &w,
                                  const std::vector<int> &h,
                                  std::vector<arma::mat>& s){

    cv::Rect image_box(0,0, img_cols, img_rows);

    CHECK_EQ(s.size(), R.size());
    CHECK_EQ(R.size(), w.size());
    CHECK_EQ(w.size(), h.size());

    for (int slice=0; slice < s.size(); slice++) {

        int r = R[slice];
        int width = w[slice];
        int height = h[slice];

        for (int x=0; x<s.at(slice).n_cols; x++) {

            for (int y=0; y<s.at(slice).n_rows; y++) {

                int x_norm = (x-center_x);
                int y_norm = (y-center_y);
                // only interested in the values in the shpere
                //if (sqrt(x_norm*x_norm + y_norm*y_norm)>r) continue;



                cv::Point top_left(x, y);
                cv::Point bottom_right(x+width, y+height);

                //continue;
                if (image_box.contains(top_left) &&
                    image_box.contains(bottom_right)) {

                    int c_x = x + floor( width / 2.0);
                    int c_y = y + floor(height / 2.0);
                    cv::Rect rect(x, y,
                                  width, height);
                    double straddle = this->computeStraddling(rect);
                    s[slice](c_y, c_x) = straddle;
                    // std::cout<< "Coordinates: " << slice <<" " << x << " "
                    //         << y <<" "<< s[slice](x,y)<<std::endl;

                }
            }
        }

    }
}


void Straddling::renormalize(std::vector<arma::mat>& s, int r){
    // renormalize the matrix s

}

arma::mat Straddling::nonMaxSuppression(const arma::mat &s, int n){
    using namespace arma;

    int W  = s.n_cols - 1;
    int H  = s.n_rows - 1;
    mat suppressed(H + 1, W + 1, fill::zeros);
    std::cout << s.n_rows << " " << s.n_cols << std::endl;
    std::cout << suppressed.n_rows << " " << suppressed.n_cols << std::endl;
    int width_n = (W - 2*n)/(n+1);
    int height_n = (H - 2*n)/(n+1);
    arma::vec w_vec = arma::linspace<arma::vec>(n, W - n, width_n);
    arma::vec h_vec = arma::linspace<arma::vec>(n, H - n, height_n);

    for (int ii = 0; ii< h_vec.size(); ii++) {
        for (int jj = 0; jj < w_vec.size(); jj++) {
            int i = h_vec[ii];
            int j = w_vec[jj];
            // initialize maximum
            int mi = i;
            int mj = j;
            // search within the block
            for (int i2 = i; i2<= min(i + n, W); i2++) {
                for (int j2 = j; j2 <= min(j + n, H ); j2++) {
                    if (s(i2,j2) > s(mi, mj)) {
                        mi = i2;
                        mj = j2;
                    }
                }

            }

            // if there is a better value in the neighborhood -> ignore
            // (mi,mj)
            for (int i2 = mi - n; i2<=min( mi + n, W); i2++) {
                if (i2>=i || i2<= i + n) {
                    continue;
                }
                for (int j2 = mj - n; j2 <= min(mj + n, H); j2++) {
                    if (j2 >=j || j2 <=j + n) {
                        continue;
                    }

                    if (s(i2,j2) > s(mi, mj)) {
                        goto failed;
                    }
                }

            }
            {
                // found maximum at (mi,mj)
                suppressed(mi,mj) = s(mi,mj);
            }
        failed:
            continue;
        }
    }

    return suppressed;
}

std::pair<std::vector<cv::Rect>, std::vector<double>> Straddling::
    nonMaxSuppression(std::vector<arma::mat> &s,
                      int n,
                      std::vector<int> &w,
                      std::vector<int> &h){

    std::vector<cv::Rect> boxes;
    std::vector<double> boxes_objness;


    for (int slice = 0; slice < s.size(); slice++) {

        n++;
        using namespace arma;

        int W  = s[slice].n_cols - 1;
        int H  = s[slice].n_rows - 1;

        int width_n = (W - 2*n)/(n+1);
        int height_n = (H - 2*n)/(n+1);
        arma::vec w_vec = arma::linspace<arma::vec>(n, W - n, width_n);
        arma::vec h_vec = arma::linspace<arma::vec>(n, H - n, height_n);

        for (int ii = 0; ii< h_vec.size(); ii++) {
            for (int jj = 0; jj < w_vec.size(); jj++) {
                int i = h_vec[ii];
                int j = w_vec[jj];
                // initialize maximum
                int mi = i;
                int mj = j;
                // search within the block
                for (int i2 = i; i2<= min(i + n, W); i2++) {
                    for (int j2 = j; j2 <= min(j + n, H ); j2++) {
                        if (s[slice](i2,j2) > s[slice](mi, mj)) {
                            mi = i2;
                            mj = j2;
                        }
                    }

                }

                // if there is a better value in the neighborhood -> ignore
                // (mi,mj)
                for (int i2 = mi - n; i2<=min( mi + n, W); i2++) {
                    if (i2>=i || i2<= i + n) {
                        continue;
                    }
                    for (int j2 = mj - n; j2 <= min(mj + n, H); j2++) {
                        if (j2 >=j || j2 <=j + n) {
                            continue;
                        }

                        if (s[slice](i2,j2) > s[slice](mi, mj)) {
                            goto failed;
                        }
                    }

                }
                {
                    if (s[slice](mi, mj)!= 0) {
                        // found maximum at (mi,mj)
                        cv::Rect local_max_box(mi, mj, w[slice], h[slice]);
                        boxes.push_back(local_max_box);
                        boxes_objness.push_back(s[slice](mi, mj));
                    }
                }
            failed:
                continue;
            }
        }

    }

    std::pair<std::vector<cv::Rect>, std::vector<double>> result =
        std::make_pair(boxes, boxes_objness);

    return result;
}

double Straddling::computeStraddling(cv::Rect &rect_big){
    double measure=0;
    // Transpose operation so that rect_big could be passed in
    // image coordinates, not matrix ones.
    //cv::Rect rect(rect_big.y, rect_big.x, rect_big.height, rect_big.width);
    cv::Rect rect = this->getInnerRect(rect_big,this->inner_threshold);
    //cv::Rect rect(rect_big.y, rect_big.x, rect_big.height, rect_big.width);
    int n=this->integrals.n_rows-1;
    int m=this->integrals.n_cols-1;

    for (int superpixel=0; superpixel<this->integrals.n_slices; superpixel++) {

        //            int A=integrals(rect.x+rect.width,
        //                            rect.y+rect.height,superpixel);
        //
        //            int B=integrals(rect.x,rect.y,superpixel);
        //
        //            int C=integrals(rect.x+rect.width,rect.y,superpixel);
        //
        //            int D=integrals(rect.x,rect.y+rect.height,superpixel);
        
        int area_superpixel_window_overlap=integrals(rect.x+rect.width,
                                                     rect.y+rect.height,
                                                     superpixel)+
            integrals(rect.x,rect.y,superpixel)-
            integrals(rect.x+rect.width,rect.y,superpixel)-
            integrals(rect.x,rect.y+rect.height,superpixel);
        
        int area_superpixel_without_window=integrals(n,m,superpixel)-
            area_superpixel_window_overlap;
        
        
        //int sum=integrals(n,m,superpixel);
        measure+=MIN(area_superpixel_window_overlap,
                     area_superpixel_without_window)/
        ((double)(rect.width*rect.height));
        
    }
    
    measure=1-measure;
    
    return measure;
}


void EdgeDensity::edgeOnCube(int img_rows,
                             int img_cols,
                             int center_x,
                             int center_y,
                             const std::vector<int> &R,
                             const std::vector<int> &w,
                             const std::vector<int> &h,
                             std::vector<arma::mat>& s){
    cv::Rect image_box(0,0, img_cols, img_rows);

    CHECK_EQ(s.size(), R.size());
    CHECK_EQ(R.size(), w.size());
    CHECK_EQ(w.size(), h.size());

    for (int slice=0; slice < s.size(); slice++) {

        int r = R[slice];
        int width = w[slice];
        int height = h[slice];

        for (int x=0; x<s.at(slice).n_cols; x++) {

            for (int y=0; y<s.at(slice).n_rows; y++) {

                int x_norm = (x-center_x);
                int y_norm = (y-center_y);
                // only interested in the values in the shpere
                //if (sqrt(x_norm*x_norm + y_norm*y_norm)>r) continue;



                cv::Point top_left(x, y);
                cv::Point bottom_right(x+width, y+height);

                //continue;
                if (image_box.contains(top_left) &&
                    image_box.contains(bottom_right)) {

                    int c_x = x + floor( width / 2.0);
                    int c_y = y + floor(height / 2.0);
                    cv::Rect rect(x, y,
                                  width, height);
                    double straddle = this->computeEdgeDensity(rect);
                    s[slice](c_y, c_x) = straddle;
                    // std::cout<< "Coordinates: " << slice <<" " << x << " "
                    //         << y <<" "<< s[slice](x,y)<<std::endl;

                }
            }
        }

    }

}

void EdgeDensity::preprocessIntegral(cv::Mat& image){
    cv::Mat edges = this->getEdges(image);
    this->computeIntegrals(edges);
}

/**
 *  Compute edges from the image using Canny edge detector
 *
 *  @param image cv::Mat image (either RGB or grayscale)
 *
 *  @return cv::Mat of type CV_8U with edges
 */
cv::Mat EdgeDensity::getEdges(cv::Mat& image){
    using namespace cv;
    
    Mat detected_edges(image.rows,image.cols,CV_8U);
    
    blur(image, detected_edges, Size(3,3));  //<-- this line cannot be deleted
    
    Canny(detected_edges,detected_edges,this->threshold_1,this->threshold_2);

    
    return detected_edges;
}


void EdgeDensity::computeIntegrals(cv::Mat &labels){
    int m=labels.cols;
    int n=labels.rows;
    
    //arma::Mat<int> edges_x(n+1,m+1,arma::fill::zeros);
    //arma::Mat<int> edges_y(n+1,m+1,arma::fill::zeros);
    
    this->edges_x.set_size(n+1, m+1);
    this->edges_x.fill(0);
    
    this->edges_y.set_size(n+1, m+1);
    this->edges_y.fill(0);
    
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
}


arma::rowvec EdgeDensity::findEdgeObjectness(std::vector<cv::Rect> &rects, int translate_x, int translate_y){

    arma::rowvec measures(rects.size(),arma::fill::zeros);
    for (int i=0; i<rects.size(); i++) {
        
        
        cv::Rect rect(rects[i].x-translate_x,rects[i].y-
                      translate_y,rects[i].width,rects[i].height);

        
        measures[i]=computeEdgeDensity(rect);
    }
    
    
    
    return measures;
    
}

double EdgeDensity::computeEdgeDensity(cv::Rect &rect){
    cv::Rect inner_rect=Straddling::getInnerRect(rect,this->inner_threshold);
    
    // calculate how many edges on the perimeter of the inner rectangle
    double measure=0;
    
    int edges_in_x=edges_x(inner_rect.y,inner_rect.x+inner_rect.width)+edges_x(inner_rect.y+inner_rect.height,inner_rect.x+inner_rect.width)-edges_x(inner_rect.y,inner_rect.x)-edges_x(inner_rect.y+inner_rect.height,inner_rect.x);
    
    int edges_in_y=edges_y(inner_rect.y+inner_rect.height,inner_rect.x)+edges_y(inner_rect.y+inner_rect.height,inner_rect.x+inner_rect.width)-edges_y(inner_rect.y,inner_rect.x)-edges_y(inner_rect.y,inner_rect.x+inner_rect.width);
    
    //        int edges_in_x=edges_x(inner_rect.x+inner_rect.width,inner_rect.y)+edges_x(inner_rect.x+inner_rect.width,inner_rect.y+inner_rect.height)-edges_x(inner_rect.x,inner_rect.y)-edges_x(inner_rect.x,inner_rect.y+inner_rect.height);
    //
    //        int edges_in_y=edges_y(inner_rect.x,inner_rect.y+inner_rect.height)+edges_y(inner_rect.x+inner_rect.width,inner_rect.y+inner_rect.height)-edges_y(inner_rect.x,inner_rect.y)-edges_y(inner_rect.x+inner_rect.width,inner_rect.y);
    
    measure=(edges_in_x+edges_in_y)/((double)(2*(inner_rect.width+inner_rect.height)));
    
    return measure;
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






























