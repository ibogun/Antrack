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

#include "Kernels/RBFKernel.h"
#include "Kernels/IntersectionKernel.h"
#include "Kernels/IntersectionKernel_fast.h"
#include "Kernels/ApproximateKernel.h"

#include "Features/RawFeatures.h"
#include "Features/Haar.h"
#include "Features/Histogram.h"

#include "Tracker/LocationSampler.h"
#include "Tracker/OLaRank_old.h"
#include "Tracker/Struck.h"

#include "Datasets/DataSetWu2013.h"
#include "Datasets/DatasetALOV300.h"
#include "Datasets/DatasetVOT2014.h"

#include <pthread.h>
#include <thread>


#ifdef _WIN32
//define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
//define something for Windows (64-bit only)
#endif
#elif __APPLE__
#include "TargetConditionals.h"

#define NUM_THREADS         10

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
#elif __unix // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif


Struck getTracker(){
    // Parameters
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params p;
    p.C          = 100;
    p.n_O        = 10;
    p.n_R        = 10;
    int nRadial  = 5;
    int nAngular = 16;
    int B        = 100;
    
    //RawFeatures* features=new RawFeatures(16);
    HistogramFeatures* features=new HistogramFeatures(4,16);
    // RBFKe
    //IntersectionKernel_fast* kernel=new IntersectionKernel_fast;
    //ApproximateKernel* kernel= new ApproximateKernel(30);
    IntersectionKernel* kernel=new IntersectionKernel;
    //Haar* features=new Haar(2);
    
    int verbose = 0;
    int display = 0;
    int m       = features->calculateFeatureDimension();
    int K       = nRadial*nAngular+1;
    
    OLaRank_old* olarank=new OLaRank_old(kernel);
    olarank->setParameters(p, B,m,K,verbose);
    
    int r_search = 30;
    int r_update = 60;
    
    bool useFilter=true;
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    LocationSampler* samplerForUpdate = new LocationSampler(r_update,nRadial,nAngular);
    LocationSampler* samplerForSearch = new LocationSampler(r_search,nRadial,nAngular);
    
    Struck tracker(olarank, features,samplerForSearch, samplerForUpdate,useFilter, display);
    
    return tracker;
}






void runTrackerOnDatasetPart(vector<pair<string, vector<string>>>& video_gt_images,Dataset* dataset,
                             int from, int to,std::string saveFolder, bool saveResults){
    
    Struck tracker=getTracker();
     std::time_t t1 = std::time(0);
    
    int frameNumber = 0;
    // paralelize this loop
    for (int videoNumber=from; videoNumber<to; videoNumber++) {
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
        
        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
        
        frameNumber+=gt_images.second.size();
        cv::Mat image=cv::imread(gt_images.second[0]);
        
        
        tracker.initialize(image, groundTruth[0]);
        
        
        for (int i=1; i<5; i++) {
            //for (int i=1; i<gt_images.second.size(); i++) {
            
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

void applyTrackerOnDataset(Dataset *dataset, std::string rootFolder, std::string saveFolder, bool saveResults){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
    
    std::time_t t1 = std::time(0);

    
    std::vector<std::thread> th;
    
    arma::rowvec bounds=arma::linspace<rowvec>(0, video_gt_images.size(),NUM_THREADS);
    
    bounds=arma::round(bounds);
    
    for (int i=0; i<NUM_THREADS-1; i++) {
        th.push_back(std::thread(runTrackerOnDatasetPart,std::ref(video_gt_images),std::ref(dataset),std::ref(bounds[i]),std::ref(bounds[i+1]),std::ref(saveFolder),std::ref(saveResults)));
    }
    
    for(auto &t : th){
        t.join();
    }
    
   
    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;
    
}


int main(int argc, const char * argv[]) {

    DataSetWu2013* wu2013=new DataSetWu2013;
    
    //DatasetALOV300* alov300=new DatasetALOV300;
    
    DatasetVOT2014* vot2014=new DatasetVOT2014;
    
    //vot2014->showVideo(vot2014RootFolder,0);
    
    //applyTrackerOnDataset(wu2013, wu2013RootFolder, wu2013SaveFolder, true);
    applyTrackerOnDataset(vot2014, vot2014RootFolder, vot2014SaveFolder, true);
    
    //tracker.applyTrackerOnVideoWithinRange(datasetWu2013, rootFolder, 0, 0, 250);
    //tracker.videoCapture();
    
    return 0;
}
