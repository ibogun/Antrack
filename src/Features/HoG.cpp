//
//  HoG.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/26/15.
//
//

#include "HoG.h"


HoG::HoG(cv::Size size_){
    
    cv::Size winSize(64,64);
    cv::Size blockSize(32,32);
    cv::Size cellSize(8,8);
    cv::Size blockStride(16,16);
    int nBins=6;
    
    cv::HOGDescriptor* d_=new cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);
    
    this->size=size_;
    this->d=d_;
}

int HoG::calculateFeatureDimension(){
    
    return (int)this->d->getDescriptorSize();
}

cv::Mat HoG::prepareImage(cv::Mat *imageIn){
    cv::Mat image=*imageIn;
    cv::Mat gray(image.rows,image.cols,CV_16S);
    
    if (image.channels()!=1){
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    }else{
        gray=image;
    }
    return gray;
}


std::string HoG::getInfo(){
    
    std::string r="HoG feature with\nwidth/height      : " +std::to_string(this->size.width)+", "+std::to_string(this->size.height)+"\n"+"Feature dimension : "+std::to_string(this->d->getDescriptorSize())+"\n";
    return r;
}

arma::mat HoG::calculateFeature(cv::Mat &processedImage, std::vector<cv::Rect> &locationsInCropped){
    
    using namespace cv;
    
    //FIXME: Inefficient extraction of the HoG features - there should be a way to do it faster
    
    int m=this->calculateFeatureDimension();
    int n=(int)locationsInCropped.size();
    
    arma::mat x(n,m,arma::fill::zeros);
    
    for (int i=0; i<n; i++) {
        
        cv::Mat cropped(processedImage,locationsInCropped[i]);
        
        //std::cout<<i<<" "<<locationsInCropped[i]<<std::endl;
        cv::resize(cropped,cropped,this->size);
        
        vector<float> descriptorsValues;
        
        this->d->compute(cropped, descriptorsValues);
        
        for (int j=0; j<m; j++) {
            x(i,j)=descriptorsValues[j];
        }
        
    }
    
    
    return x;
    
    
}

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
                                       std::vector<float>& descriptorValues,
                                       cv::Size winSize,
                                       cv::Size cellSize,
                                       int scaleFactor,
                                       double viz_factor)
{
    cv::Mat visual_image;
    
    using namespace std;
    resize(origImg, visual_image, cv::Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
    
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize;
    
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
            
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
    
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
    
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
    
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
                
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
                    
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                    
                } // for (all bins)
                
                
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
                
            } // for (all cells)
            
            
        } // for (all block x pos)
    } // for (all block y pos)
    
    
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
            
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
    
    
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
    
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
            
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
            
            rectangle(visual_image,
                      cv::Point(drawX*scaleFactor,drawY*scaleFactor),
                      cv::Point((drawX+cellSize.width)*scaleFactor,
                                (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
            
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
                
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
                
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
                
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                // to see the lines better
                
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                
                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1*scaleFactor,y1*scaleFactor),
                     cv::Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
                
            } // for (all bins)
            
        } // for (cellx)
    } // for (celly)
    
    
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    
    return visual_image;
    
}
