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


#include "Tracker/LocationSampler.h"
#include "Tracker/OLaRank_old.h"
#include "Tracker/Struck.h"

#include "Datasets/DataSetWu2013.h"
#include "Datasets/DatasetALOV300.h"
#include "Datasets/DatasetVOT2014.h"
#include "Datasets/EvaluationRun.h"

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

#define NUM_THREADS         16

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

// OPTLEX machine
//#define wu2013RootFolder    "/media/drive/UbuntuFiles/Datasets/Tracking/wu2013/"
#define wu2013RootFolder "/udrive/student/ibogun2010/Research/Data/Tracking_benchmark/"

#define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder   "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"

// OPTLEX machine
// #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"

#define wu2013SaveFolder  "/udrive/student/ibogun2010/Research/Results/wu2013/"
#define alovSaveFolder      "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder    "/media/drive/UbuntuFiles/Results/vot2014"
#elif __unix // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif





void runTrackerOnDatasetPart(vector<pair<string, vector<string>>>& video_gt_images,Dataset* dataset,
                             int from, int to,std::string saveFolder, bool saveResults, int nFrames){





    std::time_t t1 = std::time(0);

    int frameNumber = 0;
    // paralelize this loop
    for (int videoNumber=from; videoNumber<to; videoNumber++) {
        Struck tracker=Struck::getTracker();
         tracker.display=0;
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];

        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);

        frameNumber+=gt_images.second.size();
        cv::Mat image=cv::imread(gt_images.second[0]);


        tracker.initialize(image, groundTruth[0]);


        nFrames=MIN(nFrames,gt_images.second.size());

        for (int i=1; i<nFrames; i++) {

            cv::Mat image=cv::imread(gt_images.second[i]);


            tracker.track(image);
        }

        if (saveResults) {
            std::string saveFileName=saveFolder+"/"+dataset->videos[videoNumber]+".dat";



            tracker.saveResults(saveFileName);
        }

        EvaluationRun r;

        r.evaluate(groundTruth, tracker.boundingBoxes);

        std::cout<<dataset->videos[videoNumber]<<std::endl;
        std::cout<<r<<std::endl;

        //tracker.reset();


    }

    std::time_t t2 = std::time(0);
    std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    //std::cout<<"No threads: "<<(t2-t1)<<std::endl;
}


vector<EvaluationRun> applyTrackerOnDataset(Dataset* dataset, std::string rootFolder,
                                             std::string saveFolder,int frames=10){

    vector<EvaluationRun> results;

    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);

    std::time_t t1 = std::time(0);

    #pragma omp parallel for
    for (int i=0; i<video_gt_images.size(); i++) {
        Struck tracker=Struck::getTracker();
        tracker.display=0;
        int n=MIN(frames,video_gt_images[i].second.size());
        EvaluationRun r= tracker.applyTrackerOnVideoWithinRange(dataset, rootFolder,saveFolder, i, 0, n);

        results.push_back(r);
    }

    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;


    return results;
}

void applyTrackerOnDataset(Dataset *dataset, std::string rootFolder, std::string saveFolder, bool saveResults,int n_threads, int nFrames=5000){

    using namespace std;

    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);

    std::time_t t1 = std::time(0);


    std::vector<std::thread> th;

    arma::rowvec bounds=arma::linspace<rowvec>(0, video_gt_images.size(),MIN(n_threads,video_gt_images.size()));

    bounds=arma::round(bounds);

    for (int i=0; i<n_threads; i++) {
        th.push_back(std::thread(runTrackerOnDatasetPart,std::ref(video_gt_images),std::ref(dataset),std::ref(bounds[i]),std::ref(bounds[i+1]),std::ref(saveFolder),std::ref(saveResults),std::ref(nFrames)));
    }

    for(auto &t : th){
        t.join();
    }


    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
    std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;

    std::ofstream out(saveFolder+"/"+"tracker_info.txt");
    out << Struck::getTracker();
    out.close();


}


int main(int argc, const char * argv[]) {

    DataSetWu2013* wu2013=new DataSetWu2013;



    //DatasetALOV300* alov300=new DatasetALOV300;
    //
    //DatasetVOT2014* vot2014=new DatasetVOT2014;

   // std::string wuName="/udrive/student/ibogun2010/Research/Data/Tracking_benchmark/";



    //std::vector<std::pair<std::string, std::vector<std::string>>> votPrepared=vot2014->prepareDataset(vot2014RootFolder);
    std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared=wu2013->prepareDataset(wu2013RootFolder);

    Struck tracker=Struck::getTracker(false,false,false,false,false);


    //vot2014->showVideo(vot2014RootFolder,0);
    int frames=50;

    int n_threads=50;

    std::cout<<wu2013RootFolder<<std::endl;
    std::cout<<wu2013SaveFolder<<std::endl;

    //applyTrackerOnDataset(wu2013, wu2013RootFolder, wu2013SaveFolder,true,n_threads,frames);

    std::string vidName="shaking";
    int vidIndex=wu2013->vidToIndex.at(vidName);
    //tracker.display=0;


    tracker.display=2;
    EvaluationRun run= tracker.applyTrackerOnVideoWithinRange(wu2013, wu2013RootFolder,wu2013SaveFolder, vidIndex, 0, 900);


    //tracker.reset();

    //delete wu2013;
    return 0;
}
