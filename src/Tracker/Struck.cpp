//
//  Struck.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Struck.h"
#include <math.h>

void Struck::initialize(cv::Mat &image, cv::Rect &location){
    
    srand(this->seed);
    // set dimensions of the sampler
    
    
    // NOW
    int m=image.rows;
    int n=image.cols;
    
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
    updateTracker=true;
    if (this->useFilter) {
        int measurementSize=10;
        colvec x_k(measurementSize,fill::zeros);
        x_k(0)=lastLocation.x;
        x_k(1)=lastLocation.y;
        x_k(2)=lastLocation.x+lastLocation.width;
        x_k(3)=lastLocation.y+lastLocation.height;
        
        int robustConstant_b=10;
        
        int R_cov=5;
        int Q_cov=5;
        int P=3;
        
        
        filter=KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(x_k,n,m,R_cov,Q_cov,P,robustConstant_b);
        lastRectFilterAndDetectorAgreedOn=lastLocation;
        
        //filter.x_kk=x_k;
        
        
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
    
    if (pretraining) {
        this->preTraining(image, location);
    }
    
}



cv::Rect Struck::track(cv::Mat &image){
    
    std::vector<cv::Rect> locationsOnaGrid;
    locationsOnaGrid.push_back(lastLocation);
    
    // to reproduce results in the paper
    //    this->samplerForSearch->sampleOnAGrid(lastLocation, locationsOnaGrid, this->R,2);
    
    this->samplerForSearch->sampleEquiDistantMultiScale(lastLocation, locationsOnaGrid);
    //this->samplerForSearch->sampleEquiDistant(lastLocation, locationsOnaGrid);
    if (useFilter && !updateTracker) {
        this->samplerForSearch->sampleOnAGrid(lastRectFilterAndDetectorAgreedOn, locationsOnaGrid, this->R,2);
    }
    
    // to make it work faster
    //this->samplerForSearch->sampleEquiDistant(lastLocation, locationsOnaGrid);
    
    
    
    cv::Mat processedImage=this->feature->prepareImage(&image);
    
    arma::mat x=this->feature->calculateFeature(processedImage, locationsOnaGrid);
    
    arma::rowvec predictions=this->olarank->predictAll(x);
    if (this->useObjectness) {
        
        
        
        predictions=predictions-predictions.min();
        predictions=predictions/sum(predictions);
        
        //Objectne
        //arma::rowvec obj_measure
        this->weightWithStraddling(image, predictions, locationsOnaGrid, 200);
        //predictions=predictions % obj_measure;
    }
    
    
    if(this->scalePrior){
        int sigma=10;
        
        int two_sigma_squared=2*pow(sigma,2);
        
        arma::rowvec scalePriors(locationsOnaGrid.size(),arma::fill::zeros);
        
        for (int i=0; i<locationsOnaGrid.size(); i++) {
            scalePriors[i]=expf(-(sqrt(pow(locationsOnaGrid[i].width-
                                           this->lastLocation.width,2)+pow(locationsOnaGrid[i].height-this->lastLocation.height,2))/two_sigma_squared));
            
            //expf(-(pow(locationsOnaGrid[i].height-this->lastLocation.height,2)/two_sigma_squared));
        }
        
        predictions=predictions % scalePriors;
        
    }
    
    
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
        
        this->lastLocationFilter=bestLocationFilter;
        
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
    
    return lastLocation;
    
}


typedef std::pair<double,int> my_pair;

bool comparator_pair( const my_pair& l, const my_pair& r)
{ return l.first>r.first;}

void Struck::updateDebugImage(cv::Mat* canvas,cv::Mat& img, cv::Rect &bestLocation,cv::Scalar colorOfBox){
    
    int WIDTH_fitted=this->samplerForUpdate->objectWidth;  // this dimension does not change
    
    int HEIGHT_fitted=this->samplerForUpdate->objectHeight;
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
    
    
    if (useFilter) {
        cv::rectangle(plotImg,this->lastLocationFilter,cv::Scalar(255,193,0),2);
    }
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


std::pair<arma::rowvec, arma::rowvec> Struck::weightWithStraddling(cv::Mat &image, arma::rowvec &predictions,
                                  std::vector<cv::Rect>&rects,const int nSuperpixels){
    
    double inner=0.9;
    
    int objDisplay=2;
    Straddling straddle(nSuperpixels,inner,objDisplay);
    
    
    
    
    // iterate over all rects and location minimum and maximu
    
    int max_x,max_y=0;
    
    int min_x=image.cols;
    int min_y=image.rows;
    
    for (int i=0; i<rects.size(); i++) {
        
        min_x=MIN(min_x, rects[i].x);
        min_y=MIN(min_y,rects[i].y);
        max_x=MAX(max_x, rects[i].x+rects[i].width);
        max_y=MAX(max_y,rects[i].y+rects[i].height);
    }
    
    int delta=20;
    min_x=MAX(0,min_x-delta);
    min_y=MAX(0,min_y-delta);
    
    max_x=MIN(image.cols-1,max_x+delta);
    max_y=MIN(image.rows-1,max_y+delta);
    
    cv::Rect roi(min_x,min_y,max_x-min_x,max_y-min_y);
    
    cv::Mat smallImage(image,roi);
    
    //    cv::imshow("smallImage", smallImage);
    //    cv::waitKey();
    //    cv::destroyAllWindows();
    
    
    arma::mat labels=straddle.getLabels(smallImage);
    
    
    arma::rowvec obj_measure_fast=straddle.findStraddlng_fast(labels, rects,min_x,min_y);
    //arma::rowvec obj_measure_fast=straddle.findStraddling(labels, rects, min_x, min_y);
    
    
    EdgeDensity edgeDensity(0.5, 0.5, 0.9,objDisplay);
    //
    cv::Mat edges=edgeDensity.getEdges(smallImage);
    arma::rowvec edge_measure=edgeDensity.findEdgeObjectness(edges, rects, min_x, min_y);
    //  predictions=predictions % edge_measure;
    
    
    if(objDisplay){
        
        uword maxBoxStraddling,maxBoxEdges, maxBoxObjectness;
        
        obj_measure_fast.max(maxBoxStraddling);
        
        
        edge_measure.max(maxBoxEdges);
        
        arma::rowvec productObjScores=obj_measure_fast % edge_measure;
        
        
        
        productObjScores.max(maxBoxObjectness);
        
        this->lastLocationObjectness=rects[maxBoxObjectness];
        cv::Rect r=rects[maxBoxStraddling];
        cv::Rect bestStraddlingRect(r.x-min_x,r.y-min_y,r.width,r.height);
        
        r=rects[maxBoxEdges];
        
        cv::Rect bestEdgesRect(r.x-min_x,r.y-min_y,r.width,r.height);
        
        // convert gray to rgb
        cv::cvtColor(straddle.canvas, straddle.canvas, cv::COLOR_GRAY2RGB);
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2RGB);
        
        cv::rectangle(straddle.canvas, bestStraddlingRect,cv::Scalar(100,255,0),2);
        cv::rectangle(edges, bestEdgesRect,cv::Scalar(100,255,0),2);
        
        
        // find dimensions
        int separation=10;
        int w=image.cols+smallImage.cols+separation;
        int h=MAX(image.rows, 2*smallImage.rows);
        
        
        this->objectnessCanvas=cv::Mat(h,w,edges.type(),CV_RGB(0, 0, 0));
        
        cv::Rect r1(0,0,image.rows,image.cols);
        cv::Rect r2(0,image.cols+separation,smallImage.rows,smallImage.cols);
        cv::Rect r3(smallImage.rows,image.cols+separation,smallImage.rows,smallImage.cols);
        
        this->copyFromRectangleToImage(this->objectnessCanvas, image, r1, 0, cv::Vec3b(0,0,0));
        
        
        this->copyFromRectangleToImage(this->objectnessCanvas, straddle.canvas, r2, 0, cv::Vec3b(0,0,0));
        this->copyFromRectangleToImage(this->objectnessCanvas, edges, r3, 0, cv::Vec3b(0,0,0));
        
    }
    
    std::pair<arma::rowvec, arma::rowvec> objectnessMeasures;
    objectnessMeasures.first=obj_measure_fast;
    objectnessMeasures.second=edge_measure;
    
    predictions=predictions % (obj_measure_fast % edge_measure);
    
    return objectnessMeasures;
}


/**
 *  Apply function on the dataset
 *
 *  @param dataset     Dataset object
 *  @param rootFolder  root folder for the dataset
 *  @param videoNumber which video to use, -1 means use all, default=-1
 */
void Struck::applyTrackerOnVideo(Dataset *dataset,std::string rootFolder,int videoNumber){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    if(videoNumber<0 || videoNumber>=video_gt_images.size()){
        
        std::cout<<"Video number is incorrect"<<std::endl;
        return;
    }
    
    
    pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
    
    vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
    
    
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


void Struck::applyTrackerOnDataset(Dataset *dataset, std::string rootFolder, std::string saveFolder, bool saveResults){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    std::time_t t1 = std::time(0);
    
    string deleteFolderCommand="rm -r "+saveFolder;
    string createFolderCommand="mkdir "+saveFolder;
    
    int frameNumber=0;
    for (int videoNumber=0; videoNumber<video_gt_images.size(); videoNumber++) {
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
        
        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
        
        
        cv::Mat image=cv::imread(gt_images.second[0]);
        
        
        this->initialize(image, groundTruth[0]);
        
        
        for (int i=1; i<5; i++) {
            //for (int i=1; i<gt_images.second.size(); i++) {
            
            cv::Mat image=cv::imread(gt_images.second[i]);
            
            this->track(image);
        }
        
        frameNumber+=gt_images.second.size();
        
        
        if (saveResults) {
            std::string saveFileName=saveFolder+"/"+dataset->videos[videoNumber]+".dat";
            
            this->saveResults(saveFileName);
        }
        
        this->reset();
        
        
    }
    std::time_t t2 = std::time(0);
    std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    //std::cout<<"No threads: "<<(t2-t1)<<std::endl;
    
    std::ofstream out(saveFolder+"/"+"tracker_info.txt");
    out << this;
    out.close();
}


EvaluationRun Struck::applyTrackerOnVideoWithinRange(Dataset *dataset, std::string rootFolder, std::string saveFolder, int videoNumber, int frameFrom, int frameTo, bool saveResults){
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    
    pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
    
    vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
    
    frameTo=MIN(frameTo, gt_images.second.size());
    
    // delete everything which comes after frameTo
    gt_images.second.erase(gt_images.second.begin()+frameTo, gt_images.second.end());
    gt_images.second.erase(gt_images.second.begin(), gt_images.second.begin()+frameFrom);
    
    groundTruth.erase(groundTruth.begin()+frameTo, groundTruth.end());
    groundTruth.erase(groundTruth.begin(), groundTruth.begin()+frameFrom);
    
    cv::Mat image=cv::imread(gt_images.second[0]);
    
    //groundTruth[0].x=image.cols-1-groundTruth[0].width;
    //groundTruth[0].y=image.rows-1-groundTruth[0].height;
    
    //this->display=0;
    
    bool plotObjectness=true;
    
    this->initialize(image, groundTruth[0]);
    
    std::time_t t1 = std::time(0);
    for (int i=1; i<gt_images.second.size(); i++) {
        cv::Mat image=cv::imread(gt_images.second[i]);
        
        if (this->display>0) {
            cout<<"Frame # "<<i<<"/"<<gt_images.second.size()<<endl;
        }
        
        this->track(image);
        
        
        if (this->display==3) {
            
            
            
            cv::Scalar color(255,0,0);
            
            cv::Mat plotImg;
            if (useObjectness && plotObjectness) {
                plotImg=this->objectnessCanvas;
            }else{
                plotImg=image.clone();
            }
            
            cv::rectangle(plotImg, lastLocation, color,2);
            
            if (useObjectness && plotObjectness) {
                
                cv::rectangle(plotImg, lastLocationObjectness,cv::Scalar(0,204,102),0);
                
            }
            
            
            if (useFilter) {
                
                
                cv::Rect bestLocationFilter=this->lastLocationFilter;
                cv::rectangle(plotImg, bestLocationFilter,cv::Scalar(0,255,100),0);
            }
            cv::rectangle(plotImg, groundTruth[i],cv::Scalar(0,0,255),0);
            
            cv::imshow("Tracking window", plotImg);
            cv::waitKey(1);
            
            std::string filename="/Users/Ivan/Documents/Papers/Depth_exam/Depth/presentation/jpgs/"+to_string(1000+i)+".jpg";
            cv::imwrite(filename, plotImg);
            
        }
    }
    
    
    std::time_t t2 = std::time(0);
    
    if (this->display>0) {
        std::cout<<"FPS : "<<gt_images.second.size()/((t2-t1)*1.0)<<std::endl;
    }
    
    
    
    if (saveResults) {
        std::string filename=std::string(saveFolder)+"/"+dataset->videos[videoNumber]+".dat";
        this->saveResults(filename);
    }
    
    EvaluationRun run;
    
    run.evaluate(groundTruth, this->boundingBoxes);
    
    if (this->display>0) {
        std::cout<<run<<std::endl;
    }
    
    return run;
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


std::ostream& operator<<(std::ostream &strm,const  Struck &s) {
    
    using namespace std;
    
    
    
    string spacing="\n\n";
    string line="--------------------------------------------------------\n";
    
    strm<<"Structured tracker. \n"<<"========================================================\n";
    
    if (s.useFilter) {
        strm<<line<<"Robust Kalman filter parameters: \n"<<line<<s.filter;
    }
    
    
    
    strm<<line<<"Learning parameters: \n"+line<<*s.olarank;
    
    strm<<line<<"Sampling parameters: \n"+line<<"For search: \n"<<*s.samplerForSearch<<line<<"For update \n"<<*s.samplerForUpdate;
    
    
    strm<<line<<"Feature representation: \n"<<line<<s.feature->getInfo();
    
    strm<<"========================================================\n";
    return strm;
    
    
}


/**
 *  Update Struck on the first frame by sampling additional data by translation
 *  and rescaling original frame
 *
 *  @param image    first image
 *  @param location ground truth
 *  @param sampler  sampler object
 */
void Struck::preTraining(cv::Mat& image,const cv::Rect& location){
    
    
    // Plotting things
    const int WIDTH=this->lastLocation.width;
    const int HEIGHT=this->lastLocation.height;
    
    int WIDTH_fitted=WIDTH;
    int HEIGHT_fitted=HEIGHT;
    int maxDimension=60;
    double alpha=0;
    
    
    frames.insert({0,image});
    
    if (MAX(WIDTH, HEIGHT)>maxDimension) {
        
        // resize without changing aspect ratio
        if (WIDTH>HEIGHT) {
            alpha=(WIDTH*1.0)/maxDimension;
        }else{
            alpha=(HEIGHT*1.0)/maxDimension;
        }
        
        WIDTH_fitted=(int)(WIDTH/alpha);
        HEIGHT_fitted=(int)(HEIGHT/alpha);
    }
    
    // affine transformations: translation
    
    /*
     Increase the dimensions of the bounding box so that when the object is
     translated it would still fit inside of the bounding box.
     */
    
    
    std::vector<cv::Rect> locations;
    
    cv::Rect groundTruth(location);
    
    
    // so that the first frame wouldn't be mistaken with additional frames sampled
    // during pre-training
    int frameIndex=-1;
    
    
    int centerX=cvRound(location.x+location.width/(2.0));
    int centerY=cvRound(location.y+location.height/(2.0));
    
    
    
    bool preTrainingWithScales=true;
    bool preTrainingWithTranslation=true;
    
    this->frames.insert({frameIndex,image});
    
    
    if (this->display>=2) {
        cout<<"Pre-training..."<<endl;
    }
    
    
    
    int scales=3;
    double downsample=1.05;
    
    
    int minScale=-5;
    int maxScale=5;
    
    cv::Rect imageBox(0,0,image.cols,image.rows);
    
    
    
    
    std::vector<cv::Rect> groundTruthRects;
    
    // scale variations
    for (int scaleNum=minScale; scaleNum<=maxScale; scaleNum++) {
        if (scaleNum==0) {
            continue;
        }
        
        double alpha=(pow(downsample,scaleNum));
        
        
        groundTruth.width=location.width*alpha;
        groundTruth.height=location.height*alpha;
        
        groundTruth.x=(int)(centerX-groundTruth.width/2);
        groundTruth.y=(int)(centerY-groundTruth.height/2);
        
        //cout<<groundTruth<<endl;
        
        if (image.cols<=groundTruth.x+groundTruth.width) {
            groundTruth.width=image.cols-1-groundTruth.x;
        }
        
        if (image.rows<=groundTruth.y+groundTruth.height) {
            groundTruth.height=image.rows-1-groundTruth.y;
        }
        
        
        
        cv::Point topLeft(groundTruth.x, groundTruth.y);
        cv::Point bottomRight(groundTruth.x+groundTruth.width, groundTruth.y+groundTruth.height);
        
        if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
            
            groundTruthRects.push_back(groundTruth);
        }
        
    }
    
    
    int maxTranslation=10;
    int translationStep=5;
    
    // translation variations
    for (int dx=-maxTranslation; dx<=maxTranslation; dx=dx+translationStep) {
        for (int dy=-maxTranslation; dy<=maxTranslation; dy=dy+translationStep) {
            
            continue;
            
            if (dx==0 && dy==0) {
                continue;
            }
            
            //continue;
            
            //cout<<"Current (dx,dy): "<<dx<<" "<<dy<<endl;
            // get the bounding box
            groundTruth.x=location.x+dx;
            groundTruth.y=location.y+dy;
            groundTruth.width=location.width;
            groundTruth.height=location.height;
            
            if (image.cols<=groundTruth.x+groundTruth.width) {
                groundTruth.width=image.cols-1-groundTruth.width;
            }
            
            if (image.rows<=groundTruth.y+groundTruth.height) {
                groundTruth.height=image.rows-1-groundTruth.height;
            }
            
            cv::Point topLeft(groundTruth.x, groundTruth.y);
            cv::Point bottomRight(groundTruth.x+groundTruth.width, groundTruth.y+groundTruth.height);
            
            if (imageBox.contains(topLeft) && imageBox.contains(bottomRight)) {
                
                groundTruthRects.push_back(groundTruth);
            }
            
            
        }
    }
    
    int frameTracked=-groundTruthRects.size();
    
    if (display==2){
        
        this->allocateCanvas(image);
        this->frames.insert({framesTracked,image});
        this->updateDebugImage(&this->canvas, image, this->lastLocation, cv::Scalar(250,0,0));
    }
    
    cv::Mat processedImage=this->feature->prepareImage(&image);
    for (int i=0; i<groundTruthRects.size(); i++) {
        
        std::vector<cv::Rect> locationsOnPolarPlane;
        
        cv::Rect gt_rect=groundTruthRects[i];
        locationsOnPolarPlane.push_back(gt_rect);
        
        this->samplerForUpdate->sampleEquiDistant(gt_rect, locationsOnPolarPlane);
        
        // calculate features
        arma::mat x_update=feature->calculateFeature(processedImage, locationsOnPolarPlane);
        arma::mat y_update=this->feature->reshapeYs(locationsOnPolarPlane);
        olarank->process(x_update, y_update, 0, frameTracked);
        
        if (display==2){
            this->frames.insert({frameTracked,image});
            
            this->updateDebugImage(&this->canvas, image, gt_rect, cv::Scalar(250,0,0));
            
            
        }
        
        frameTracked++;
        
    }
    
    
    
    
}



Struck Struck::getTracker(){
    
    // Parameters
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params p;
    p.C                 = 100;
    p.n_O               = 10;
    p.n_R               = 10;
    int nRadial         = 5;
    int nAngular        = 16;
    int B               = 100;
    
    int nRadial_search  = 12;
    int nAngular_search = 30;
    
    //RawFeatures* features=new RawFeatures(16);
    cv::Size size(64,64);
    
    //HoG* features=new HoG(size);
    
    
    
    HistogramFeatures* features=new HistogramFeatures(4,16);
    // RBFKe
    IntersectionKernel_fast* kernel=new IntersectionKernel_fast;
    //ApproximateKernel* kernel= new ApproximateKernel(30);
    //IntersectionKernel* kernel=new IntersectionKernel;
    
    //RBFKernel* kernel=new RBFKernel(0.2);
    
    //HoGandRawFeatures* features=new HoGandRawFeatures(size,16);
    //LinearKernel* kernel=new LinearKernel;
    
    
    //    Haar* f1=new Haar(2);
    //    HistogramFeatures* f2=new HistogramFeatures(2,16);
    //    IntersectionKernel* k2=new IntersectionKernel;
    //    RBFKernel* k1=new RBFKernel(0.2);
    //
    //    std::vector<Kernel*> ks;
    //
    //    ks.push_back(k1);
    //    ks.push_back(k2);
    //
    //    std::vector<Feature*>    fs;
    //
    //    fs.push_back(f1);
    //    fs.push_back(f2);
    //
    //    MultiKernel* kernel=new MultiKernel(ks,fs);
    //    MultiFeature* features=new MultiFeature(fs);
    
    
    int verbose = 0;
    int display = 2;
    int m       = features->calculateFeatureDimension();
    
    OLaRank_old* olarank=new OLaRank_old(kernel);
    olarank->setParameters(p, B,m,verbose);
    
    int r_search = 30;
    int r_update = 60;
    
    
    bool pretraining   = true;
    bool useFilter     = true;
    bool useObjectness = true;
    bool scalePrior    = true;
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    LocationSampler* samplerForUpdate = new LocationSampler(r_update,nRadial,nAngular);
    LocationSampler* samplerForSearch = new LocationSampler(r_search,nRadial_search,nAngular_search);
    
    Struck tracker(olarank, features,samplerForSearch, samplerForUpdate,
                   useObjectness,scalePrior, useFilter,pretraining, display);
    
    
    int measurementSize=10;
    arma::colvec x_k(measurementSize,fill::zeros);
    x_k(0)=0;
    x_k(1)=0;
    x_k(2)=0;
    x_k(3)=0;
    
    int robustConstant_b=10;
    
    int R_cov=5;
    int Q_cov=5;
    int P=3;
    
    
    
    KalmanFilter_my filter=KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(x_k,0,0,R_cov,Q_cov,P,robustConstant_b);
    //KalmanFilter_my filter=KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(x_k,0,0,R_cov,Q_cov,P,robustConstant_b);
    //KalmanFilter_my filter=KalmanFilterGenerator::generateConstantAccelerationFilter(x_k,0,0,R_cov,Q_cov,P,robustConstant_b);
    tracker.setFilter(filter);
    
    return tracker;
    
    
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
        
        ofstream myfile;
        
        myfile.open(filename);
        
        for (int i=0; i<boundingBoxes.size(); i++) {
            myfile<<boundingBoxes[i].x<<","<<boundingBoxes[i].y<<","
            <<boundingBoxes[i].width<<","<<boundingBoxes[i].height<<"\n";
        }
        myfile.close();
        //boxes.save(filename,arma::csv_ascii);
    }
    
}