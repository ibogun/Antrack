//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <algorithm>

#include "Kernels/RBFKernel.h"
#include "Kernels/IntersectionKernel.h"
#include "Kernels/IntersectionKernel_fast.h"
#include "Kernels/ApproximateKernel.h"
#include "Kernels/LinearKernel.h"

#include "Features/RawFeatures.h"
#include "Features/Haar.h"
#include "Features/Histogram.h"
#include "Features/HoG.h"
#include "Features/HoGandRawFeatures.h"

#include "Tracker/LocationSampler.h"
#include "Tracker/OLaRank_old.h"
#include "Tracker/Struck.h"

#include "Datasets/DataSetWu2013.h"
#include "Datasets/DatasetALOV300.h"
#include "Datasets/DatasetVOT2014.h"

#include "Superpixels/SuperPixels.h"

#include <pthread.h>
#include <thread>

#include <fstream>



#ifdef _WIN32
//define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
//define something for Windows (64-bit only)
#endif
#elif __APPLE__
#include "TargetConditionals.h"

#define NUM_THREADS         8

#define wu2013RootFolder    "/Users/Ivan/Files/Data/Tracking_benchmark/"
#define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder   "/Users/Ivan/Files/Data/vot2014/"

#define wu2013SaveFolder    "/Users/Ivan/Files/Results/Tracking/wu2013"
#define alovSaveFolder      "/Users/Ivan/Files/Results/Tracking/alov300"
#define vot2014SaveFolder    "/Users/Ivan/Files/Results/Tracking/vot2014"

#if TARGET_IPHONE_SIMULATOR
// iOS Simulator
#elif TARGET_OS_IPHONE
// iOS device
#elif TARGET_OS_MAC
// Other kinds of Mac OS
#else
// Unsupported platform
#endif
#elif __linux
// linux
#define NUM_THREADS         16

#define wu2013RootFolder    "/media/drive/UbuntuFiles/Datasets/Tracking/wu2013/"
#define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder   "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"

#define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"
#define alovSaveFolder      "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder    "/media/drive/UbuntuFiles/Results/vot2014"
#elif __unix // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif


Struck getTracker(){
    // Parameters
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params p;
    p.C                 = 100;
    p.n_O               = 10;
    p.n_R               = 10;
    int nRadial         = 5;
    int nAngular        = 16;
    int B               = 33;
    
    int nRadial_search  = 12;
    int nAngular_search = 30;
    
    RawFeatures* features=new RawFeatures(16);
    cv::Size size(64,64);
    
    //HoG* features=new HoG(size);
    
    
    
    //HistogramFeatures* features=new HistogramFeatures(4,16);
    // RBFKe
    //IntersectionKernel_fast* kernel=new IntersectionKernel_fast;
    //ApproximateKernel* kernel= new ApproximateKernel(30);
    //IntersectionKernel* kernel=new IntersectionKernel;
    
    //RBFKernel* kernel=new RBFKernel(0.2);
    
    //HoGandRawFeatures* features=new HoGandRawFeatures(size,16);
    LinearKernel* kernel=new LinearKernel;
    
    
    //Haar* features=new Haar(2);
    
    int verbose = 0;
    int display = 1;
    int m       = features->calculateFeatureDimension();
    
    OLaRank_old* olarank=new OLaRank_old(kernel);
    olarank->setParameters(p, B,m,verbose);
    
    int r_search = 30;
    int r_update = 60;
    
    bool useFilter=false;
    bool useObjectness=true;
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    LocationSampler* samplerForUpdate = new LocationSampler(r_update,nRadial,nAngular);
    LocationSampler* samplerForSearch = new LocationSampler(r_search,nRadial_search,nAngular_search);
    
    Struck tracker(olarank, features,samplerForSearch, samplerForUpdate,useObjectness, useFilter, display);
    
    
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
    
    
    //KalmanFilter_my filter=KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(x_k,0,0,R_cov,Q_cov,P,robustConstant_b);
    KalmanFilter_my filter=KalmanFilterGenerator::generateConstantVelocityWithScaleFilter(x_k,0,0,R_cov,Q_cov,P,robustConstant_b);
    
    tracker.setFilter(filter);
    
    return tracker;
}






void runTrackerOnDatasetPart(vector<pair<string, vector<string>>>& video_gt_images,Dataset* dataset,
                             int from, int to,std::string saveFolder, bool saveResults, bool fullDataset){
    
    Struck tracker=getTracker();
    
    tracker.display=0;
    
    std::time_t t1 = std::time(0);
    
    int frameNumber = 0;
    // paralelize this loop
    for (int videoNumber=from; videoNumber<to; videoNumber++) {
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
        
        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
        
        frameNumber+=gt_images.second.size();
        cv::Mat image=cv::imread(gt_images.second[0]);
        
        
        tracker.initialize(image, groundTruth[0]);
        
        
        int nFrames=10;
        if (fullDataset) {
            nFrames=gt_images.second.size();
            
        }
        
        for (int i=1; i<nFrames; i++) {
            
            cv::Mat image=cv::imread(gt_images.second[i]);
            
            tracker.track(image);
        }
        
        if (saveResults) {
            std::string saveFileName=saveFolder+"/"+dataset->videos[videoNumber]+".dat";
            
            tracker.saveResults(saveFileName);
        }
        
        tracker.reset();
        
        
    }
    
    std::time_t t2 = std::time(0);
    std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    //std::cout<<"No threads: "<<(t2-t1)<<std::endl;
}

void applyTrackerOnDataset(Dataset *dataset, std::string rootFolder, std::string saveFolder, bool saveResults,bool fullDataset){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    std::time_t t1 = std::time(0);
    
    
    std::vector<std::thread> th;
    
    arma::rowvec bounds=arma::linspace<rowvec>(0, video_gt_images.size(),NUM_THREADS);
    
    bounds=arma::round(bounds);
    
    for (int i=0; i<NUM_THREADS-1; i++) {
        th.push_back(std::thread(runTrackerOnDatasetPart,std::ref(video_gt_images),std::ref(dataset),std::ref(bounds[i]),std::ref(bounds[i+1]),std::ref(saveFolder),std::ref(saveResults),std::ref(fullDataset)));
    }
    
    for(auto &t : th){
        t.join();
    }
    
    
    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;
    
    std::ofstream out(saveFolder+"/"+"tracker_info.txt");
    out << getTracker();
    out.close();
    
    
}


int main(int argc, const char * argv[]) {
    
    DataSetWu2013* wu2013=new DataSetWu2013;
    
    //DatasetALOV300* alov300=new DatasetALOV300;
    //
    DatasetVOT2014* vot2014=new DatasetVOT2014;
    

    Struck tracker=getTracker();
    //vot2014->showVideo(vot2014RootFolder,0);
    
    applyTrackerOnDataset(wu2013, wu2013RootFolder, wu2013SaveFolder, true,true);
    //applyTrackerOnDataset(vot2014, vot2014RootFolder, vot2014SaveFolder, true,false);
    
    //Struck tracker=getTracker();
    
    
    //cout<<tracker<<endl;
    //tracker.applyTrackerOnVideoWithinRange(wu2013, wu2013RootFolder, 12, 0, 250);
    //tracker.applyTrackerOnVideoWithinRange(vot2014, vot2014RootFolder, 20, 0, 550);
    //tracker.videoCapture();
    
    return 0;
}
