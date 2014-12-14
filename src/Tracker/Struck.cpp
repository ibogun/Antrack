//
//  Struck.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Struck.h"


void Struck::initialize(cv::Mat &image, cv::Rect &location){
    
    
    // set dimensions of the sampler
    int n=image.rows;
    int m=image.cols;
    
    this->samplerForSearch->setDimensions(n, m,location.height,location.width);
    this->samplerForUpdate->setDimensions(n, m,location.height,location.width);
    
    this->boundingBoxes.push_back(location);
    lastLocation=location;
    // sample in polar coordinates first
    std::vector<cv::Rect> locations;
    
    // add ground truth
    locations.push_back(location);
    this->samplerForUpdate->sampleEquiDistant(location, locations);
    
    cv::Mat processedImage=this->feature->prepareImage(&image);
    arma::mat x=this->feature->calculateFeature(processedImage, locations);
    arma::mat y=this->feature->reshapeYs(locations);
    this->olarank->initialize(x, y, 0, framesTracked);
    
    
    // create filter, if chosen
    
    if (this->useFilter) {
        int measurementSize=6;
        colvec x_k(measurementSize,fill::zeros);
        x_k(0)=lastLocation.x;
        x_k(1)=lastLocation.y;
        x_k(2)=lastLocation.x+lastLocation.width;
        x_k(3)=lastLocation.y+lastLocation.height;
        
        int robustConstant_b=10;
        
        int R_cov=5;
        int Q_cov=5;
        int P=3;
        filter=KalmanFilterGenerator::generateConstantVelocityFilter(x_k,n,m,R_cov,Q_cov,P,robustConstant_b);
        lastRectFilterAndDetectorAgreedOn=lastLocation;
        updateTracker=true;
    }
    
    // add images, in case we want to show support vectors
    
    if (display==1) {
        cv::Scalar color(255,0,0);
        cv::Mat plotImg=image.clone();
        cv::rectangle(plotImg, lastLocation, color,2);
        cv::imshow("Tracking window", plotImg);
        cv::waitKey(1);
        
    }else if (display==2){
        
        this->allocateCanvas(image);
        this->frames.insert({framesTracked,image});
        this->updateDebugImage(&this->canvas, image, this->lastLocation, cv::Scalar(250,0,0));
    }
    
    framesTracked++;
    
}



void Struck::track(cv::Mat &image){

    std::vector<cv::Rect> locationsOnaGrid;
    locationsOnaGrid.push_back(lastLocation);
    
    // to reproduce results in the paper
    this->samplerForSearch->sampleOnAGrid(lastLocation, locationsOnaGrid, this->R,2);
    
    if (useFilter && !updateTracker) {
        this->samplerForSearch->sampleOnAGrid(lastRectFilterAndDetectorAgreedOn, locationsOnaGrid, this->R,2);
    }
    
    // to make it work faster
    //this->samplerForSearch->sampleEquiDistant(lastLocation, locationsOnaGrid);
    
    cv::Mat processedImage=this->feature->prepareImage(&image);
    
    arma::mat x=this->feature->calculateFeature(processedImage, locationsOnaGrid);
    
    arma::rowvec predictions=this->olarank->predictAll(x);
    
    uword groundTruth;
    predictions.max(groundTruth);
    
    cv::Rect bestLocationDetector=locationsOnaGrid[groundTruth];
    /**
     Filter business
     **/
    cv::Rect bestLocationFilter;
    if (useFilter) {

        arma::colvec z_k(4,arma::fill::zeros);
        z_k<<bestLocationDetector.x<<bestLocationDetector.y<<bestLocationDetector.width<<bestLocationDetector.height<<endr;
        z_k(2)+=z_k(0);
        z_k(3)+=z_k(1);
        
        // make a prediction using filter
        arma::colvec x_k=filter.predict(z_k);
        
        bestLocationFilter=filter.getBoundingBox(this->lastLocation.width, this->lastLocation.height, x_k);
        
        double overlap=(bestLocationFilter&bestLocationDetector).area()/(double((bestLocationDetector |bestLocationFilter).area()));
        
        if (overlap>0.5) {
            updateTracker=true;
            
            lastRectFilterAndDetectorAgreedOn=bestLocationDetector;
            filter.setB(10);
        }else{
            
            
            updateTracker=false;
            
            filter.setB(5);
        }
        filter.predictAndCorrect(z_k);
    }
    
    
    
    /**
     Final decision on the best bounding box
     **/
    
    // add predicted location  to the results
    this->boundingBoxes.push_back(bestLocationDetector);
    lastLocation=bestLocationDetector;
    
    /**
     Tracker Update
     **/
    
    if (updateTracker) {

        
        // sample for updating the tracker
        std::vector<cv::Rect> locationsOnPolarPlane;
        locationsOnPolarPlane.push_back(lastLocation);
        
        this->samplerForUpdate->sampleEquiDistant(lastLocation, locationsOnPolarPlane);
        
        // calculate features
        arma::mat x_update=feature->calculateFeature(processedImage, locationsOnPolarPlane);
        arma::mat y_update=this->feature->reshapeYs(locationsOnPolarPlane);
        olarank->process(x_update, y_update, 0, framesTracked);
        
    }
    /**
     End of tracker udpate
     **/
    
    if (display==1) {
        cv::Scalar color(255,0,0);
        cv::Mat plotImg=image.clone();
        cv::rectangle(plotImg, lastLocation, color,2);
        
        if (useFilter) {
            cv::rectangle(plotImg, bestLocationFilter,cv::Scalar(0,255,100),0);
        }
        
        cv::imshow("Tracking window", plotImg);
        cv::waitKey(1);
        
    }else if (display==2){
        this->frames.insert({framesTracked,image});
        
        this->updateDebugImage(&this->canvas, image, this->lastLocation, cv::Scalar(250,0,0));
    }
    
    framesTracked++;
    
}


typedef std::pair<double,int> my_pair;

bool comparator_pair( const my_pair& l, const my_pair& r)
{ return l.first>r.first;}

void Struck::updateDebugImage(cv::Mat* canvas,cv::Mat& img, cv::Rect &bestLocation,cv::Scalar colorOfBox){
    
    int WIDTH_fitted=bestLocation.width;
    int HEIGHT_fitted=bestLocation.height;
    using namespace std;
    // 1) get all support betas into the vector
    vector<my_pair> beta;
    vector<cv::Mat> bb_imgs;
    
    
    
    int imgsPerRow=10;
    int imgsPerColumn=10;
    
    int idx=0;
    for (int i=0; i<this->olarank->S.size(); i++) {
        
        int frameNumber=this->olarank->S[i]->frameNumber;
        
        //        if (frameNumber<0) {
        //            frameNumber=0;
        //        }
        
        cv::Mat imageToUse=this->frames.at(frameNumber);
        
        for (int j=0; j<this->olarank->S[i]->x->n_rows; j++) {
            
            if((*this->olarank->S[i]->beta)(j)!=0){
                
                arma::mat loc=(*this->olarank->S[i]->y).row(j);
                cv::Rect rect(loc(1),loc(2),loc(3),loc(4));
                
                cv::Mat img_sp(imageToUse,rect);
                
                // cv::imshow("a",img_sp);
                //cv::waitKey();
                // resize image
                
                cv::Mat img_sp_resized(WIDTH_fitted,HEIGHT_fitted,img_sp.type());
                
                cv::resize(img_sp, img_sp_resized,cv::Size(WIDTH_fitted,HEIGHT_fitted));
                //cv::imshow("B", img_sp_resized);
                bb_imgs.push_back(img_sp_resized);
                
                auto t=std::make_pair((*this->olarank->S[i]->beta)(j),idx);
                beta.push_back(t);
                idx+=1;
            }
            
        }
    }
    
    std::sort(beta.begin(), beta.end(), comparator_pair);
    
    int I=0;
    int J=0;
    
    std::cout<<"Total support vectors: "<<beta.size()<<std::endl;
    for (int i=0; i<beta.size(); i++) {
        
        std::cout<<"Current sv (idx,value) : "<<i<<" "<< beta[i].first<<std::endl;
        if (I>=imgsPerRow) {
            I=0;
            J++;
        }
        
        
        cv::Rect position(J*HEIGHT_fitted,I*WIDTH_fitted,HEIGHT_fitted,WIDTH_fitted);
        
        cv::Vec3b color;
        if (beta[i].first>0) {
            //green color
            color[1]=255;
            color[2]=0;
        }else{
            //red color
            color[0]=0;
            color[1]=0;
            color[2]=255;
            
        }
        
        copyFromRectangleToImage(*canvas,bb_imgs[beta[i].second],position,2,color);
        I++;
        
    }
    
    cv::Mat plotImg=img.clone();
    cv::rectangle(plotImg, bestLocation, colorOfBox,2);
    
    cv::Rect position(0,imgsPerColumn*WIDTH_fitted+50,plotImg.rows,plotImg.cols);
    
    cv::Vec3b color(0,0,0);
    copyFromRectangleToImage(*canvas,plotImg,position,0,color);
    
    cv::Size resolution(1100,600);
    
    cv::Mat finalImageToShow;
    
    cv::resize(*canvas, finalImageToShow, resolution);
    //cv::imshow("Support vectors", *canvas);
    cv::imshow("Support vectors", finalImageToShow);
    cv::waitKey(1);
    
    // delete whatever is not used anymore
    
    for (auto it=this->frames.begin(); it!=this->frames.end(); ++it) {
        
        //        if (it->first==0) {
        //            continue;
        //        }
        
        bool frameExists=false;
        for (int i=0; i<this->olarank->S.size(); i++) {
            
            if(it->first==this->olarank->S[i]->frameNumber){
                frameExists=true;
            }
            
        }
        
        if (!frameExists) {
            this->frames.erase(it->first);
        }
        
    }
}


/**
 *  Apply function on the dataset
 *
 *  @param dataset     Dataset object
 *  @param rootFolder  root folder for the dataset
 *  @param videoNumber which video to use, -1 means use all, default=-1
 */
void Struck::applyTrackerOnDataset(Dataset *dataset,std::string rootFolder,int videoNumber){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    if (videoNumber<0) {
        
        // code to apply tracker on the whole dataset
        
    } else if(videoNumber<video_gt_images.size()){
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
        
        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
        
        cv::Mat image=cv::imread(gt_images.second[0]);
        
        
        this->initialize(image, groundTruth[0]);
        
        for (int i=1; i<gt_images.second.size(); i++) {
            cv::Mat image=cv::imread(gt_images.second[i]);
            
            this->track(image);
        }
    }
    
    
}


void Struck::applyTrackerOnVideoWithinRange(Dataset *dataset, std::string rootFolder, int videoNumber, int frameFrom, int frameTo){
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    
    pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
    
    vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
    
    
    // delete everything which comes after frameTo
    gt_images.second.erase(gt_images.second.begin()+frameTo, gt_images.second.end());
    gt_images.second.erase(gt_images.second.begin(), gt_images.second.begin()+frameFrom);
    
    groundTruth.erase(groundTruth.begin()+frameTo, groundTruth.end());
    groundTruth.erase(groundTruth.begin(), groundTruth.begin()+frameFrom);
    
    cv::Mat image=cv::imread(gt_images.second[0]);
    
    
    this->initialize(image, groundTruth[0]);
    
    std::time_t t1 = std::time(0);
    for (int i=1; i<gt_images.second.size(); i++) {
        cv::Mat image=cv::imread(gt_images.second[i]);
        
        this->track(image);
    }
    std::time_t t2 = std::time(0);
    
    std::cout<<"Frames per second: "<<gt_images.second.size()/(1.0*(t2-t1))<<std::endl;
}



void Struck::copyFromRectangleToImage(cv::Mat& canvas,cv::Mat& image,cv::Rect rect,int step,cv::Vec3b color){
    
    int n=canvas.rows;
    int m=canvas.cols;
    
    for (int i = 0; i < rect.width; ++i) {
        for (int j = 0; j < rect.height; ++j) {
            if (rect.x+i<n && rect.y+j<m) {
                if((i<=step-1||(rect.width-i<=step))||(j<=step-1||(rect.height-j<=step))){
                    if(image.channels()==1){
                        uchar b(255);
                        canvas.at<uchar>(rect.x+i,rect.y+j)=b;
                    }else{
                        //cv::Vec3b b(100,100,100);
                        canvas.at<cv::Vec3b>(rect.x+i,rect.y+j)=color;
                    }
                }else{
                    
                    if(image.channels()==1){
                        canvas.at<uchar>(rect.x+i,rect.y+j)=image.at<uchar>(i-step,j-step);
                    }else{
                        //cout<<image.channels()<<endl;
                        canvas.at<cv::Vec3b>(rect.x+i,rect.y+j)=image.at<cv::Vec3b>(i-step,j-step);
                    }
                    
                }
                
            }
            
        }
    }
    
}

void Struck::allocateCanvas(cv::Mat& img1){
    
    int imgsPerRow=10;
    int imgsPerColumn=10;
    
    // bounding box dimensions
    const int WIDTH_fitted=this->lastLocation.width;
    const int HEIGHT_fitted=this->lastLocation.height;
    
    
    cv::Mat* newCanvas=new cv::Mat (MAX(imgsPerRow*HEIGHT_fitted,img1.rows),imgsPerColumn*WIDTH_fitted+50+img1.cols,img1.type(),CV_RGB(0, 0, 0));
    
    canvas=*newCanvas;
    
}


void Struck::videoCapture(){
    
    using namespace cv;
    using namespace std;
    
    VideoCapture stream(0);   //0 is the id of video device.0 if you have only one camera.
    
    if (!stream.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera";
    }
    
    int idx=0;
    
    
    bool firstFrame=true;
    
    cv::Rect rect(0,0,80,120);
    
    cv::Size size(640,480);
    while (true) {
        cv::Mat cameraFrame;
        stream.read(cameraFrame);
        cv::resize(cameraFrame, cameraFrame, size);
        
        if (firstFrame) {
            firstFrame=false;
            
            int n=cameraFrame.rows;
            int m=cameraFrame.cols;
            cout<<"Image size: "<<n<<"x"<<m<<endl;
            rect.x=m/2-rect.width/2;
            rect.y=n/2-rect.height/2;
        }
        
        // draw rectangle
        cv::rectangle(cameraFrame,rect,cv::Scalar(255,0,0),2);
        
        // add text
        cv::putText(cameraFrame, "press i to start tracking", cv::Point(rect.x-10,50), 0, 1, cv::Scalar(0,255,0));
        //cout<<"Frame # "<<idx<<endl;
        idx++;
        imshow("cam", cameraFrame);
        
        if (waitKey(30) >= 0)
            break;
    }
    
    cv::Mat cameraFrame;
    stream.read(cameraFrame);
    cv::resize(cameraFrame, cameraFrame, size);
    waitKey(1);
    this->initialize(cameraFrame, rect);
    
    for (int i=0; i<250; i++) {
        cv::Mat cameraFrame;
        stream.read(cameraFrame);
        cv::resize(cameraFrame, cameraFrame, size);
        this->track(cameraFrame);
    }
}

void Struck::saveResults(string filename){
    
    if (this->boundingBoxes.size()==0) {
        std::cout<<"Run the tracker first"<<std::endl;
        return;
    }else{
        
        
        arma::mat boxes((int)this->boundingBoxes.size(),4,arma::fill::zeros);
        
        cv::Rect b;
        for (int i=0; i<this->boundingBoxes.size(); ++i) {
            b=this->boundingBoxes[i];
            boxes(i,0)=b.x;
            boxes(i,1)=b.y;
            boxes(i,2)=b.width;
            boxes(i,3)=b.height;
        }
        
        boxes.save(filename,arma::raw_ascii);
    }
    
}