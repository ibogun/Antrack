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
#include <omp.h>

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



// #ifdef _WIN32
// //define something for Windows (32-bit and 64-bit, this part is common)
// #ifdef _WIN64
// //define something for Windows (64-bit only)
// #endif
// #elif __APPLE__
// #include "TargetConditionals.h"
//
// #define NUM_THREADS         16
//
// #define wu2013RootFolder    "/Users/Ivan/Files/Data/Tracking_benchmark/"
// #define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
// #define vot2014RootFolder   "/Users/Ivan/Files/Data/vot2014/"
//
// #define wu2013SaveFolder    "/Users/Ivan/Files/Results/Tracking/wu2013"
// #define alovSaveFolder      "/Users/Ivan/Files/Results/Tracking/alov300"
// #define vot2014SaveFolder    "/Users/Ivan/Files/Results/Tracking/vot2014"
//
// #if TARGET_IPHONE_SIMULATOR
// // iOS Simulator
// #elif TARGET_OS_IPHONE
// // iOS device
// #elif TARGET_OS_MAC
// // Other kinds of Mac OS
// #else
// // Unsupported platform
// #endif
// #elif __linux
// // linux
// #define NUM_THREADS         16
//
// // OPTLEX machine
// //#define wu2013RootFolder    "/media/drive/UbuntuFiles/Datasets/Tracking/wu2013/"
// #define wu2013RootFolder "/udrive/student/ibogun2010/Research/Data/Tracking_benchmark/"
//
// #define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
// #define vot2014RootFolder   "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"
//
// // OPTLEX machine
// // #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"
//
// #define wu2013SaveFolder  "/udrive/student/ibogun2010/Research/Results/wu2013/"
// #define alovSaveFolder      "/media/drive/UbuntuFiles/Results/alov300"
// #define vot2014SaveFolder    "/media/drive/UbuntuFiles/Results/vot2014"
// #elif __unix // all unices not caught above
// // Unix
// #elif __posix
// // POSIX
// #endif
//
//
// EvaluationRun applyTracker(vector<pair<string, vector<string>>> video_gt_images, std::string rootFolder,
//                                              std::string saveFolder,int i, int n=10){
//
//
// 										         Struck tracker=Struck::getTracker();
// 										         tracker.display=0;
// 										         int n=MIN(frames,video_gt_images[i].second.size());
// 										         EvaluationRun r= tracker.applyTrackerOnVideoWithinRange(dataset, rootFolder,saveFolder, i, 0, n);
//
// 												 return r;
//                                              }
//
//
//
//
//
//
// void applyTrackerOnDatasetParralel(Dataset* dataset, std::string rootFolder,
//                                              std::string saveFolder,int frames=10){
//
//     vector<EvaluationRun> results;
//
//     vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
//
//     std::time_t t1 = std::time(0);
//
//     using namespace std;
//     /*
//     Example of the parallel loop in openmp
//
//     int i;
//     double xdoty;
//
//     xdoty = 0.0;
//
//   # pragma omp parallel \
//     shared ( n, x, y ) \
//     private ( i )
//
//   # pragma omp for reduction ( + : xdoty )
//
//
//     for ( i = 0; i < n; i++ )
//     {
//       xdoty = xdoty + x[i] * y[i];
//     }
//
//     return xdoty;
//
//
//
//     */
//
//
//       cout << "\n";
//       cout << "Data set evaluation \n";
//       cout << "  C++/OpenMP version\n";
//       cout << "\n";
//       cout << "  A program which evaluates tracking dataset in parallel.\n";
//
//       cout << "\n";
//       cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
//       cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
//
//     double average_precision=0;
//     double average_overlap  =0;
//     double average_precision_thresholded=0;
//     double average_overlap_thresholded =0;
//
//     int totalVideos=16;
//     int i=0;
//
//     #pragma omp parallel shared(totalVideos,rootFolder,saveFolder,video_gt_images) private(i)
//     #pragma omp for reduction(+:average_overlap,average_overlap_thresholded,average_precision,average_precision_thresholded)
//     for (i=0; i<totalVideos; i++) {
//
//       //for (int i=0; i<video_gt_images.size(); i++) {
//         //Struck tracker=Struck::getTracker();
//         //tracker.display=0;
//         int n=MIN(frames,video_gt_images[i].second.size());
//         //EvaluationRun r= tracker.applyTrackerOnVideoWithinRange(dataset, rootFolder,saveFolder, i, 0, n);
//
//
// 		EvaluationRun r = applyTracker(video_gt_images,rootFolder,saveFolder,i,0,n);
//
//         average_overlap=average_overlap+r.overlap_area;
//         average_overlap_thresholded=average_overlap_thresholded+r.overlap_half;
//
//         average_precision=average_precision+r.precision_area;
//         average_precision_thresholded=average_precision_thresholded+r.precision_20;
//
//     }
//
//     std::time_t t2 = std::time(0);
//     //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
//
//     std::cout<<"Overlap  : "<<average_overlap_thresholded/totalVideos<<" / "<<average_overlap/totalVideos<<std::endl;
//     std::cout<<"Precision: "<<average_precision_thresholded/totalVideos<<" / "<<average_precision/(totalVideos*50)<<" / "<<average_precision/totalVideos<<std::endl;
//     std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;
//
//
// }
//
//
// void applyTrackerOnDatasetSequential(Dataset* dataset, std::string rootFolder,
//                                              std::string saveFolder,int frames=10){
//
//     vector<EvaluationRun> results;
//
//     vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);
//
//     std::time_t t1 = std::time(0);
//
//     using namespace std;
//     /*
//     Example of the parallel loop in openmp
//
//     int i;
//     double xdoty;
//
//     xdoty = 0.0;
//
//   # pragma omp parallel \
//     shared ( n, x, y ) \
//     private ( i )
//
//   # pragma omp for reduction ( + : xdoty )
//
//
//     for ( i = 0; i < n; i++ )
//     {
//       xdoty = xdoty + x[i] * y[i];
//     }
//
//     return xdoty;
//
//
//
//     */
//
//
//       cout << "\n";
//       cout << "Data set evaluation \n";
//       cout << "  C++/sequential version\n";
//       cout << "\n";
//       cout << "  A program which evaluates tracking dataset sequentially.\n";
//
//     double average_precision=0;
//     double average_overlap  =0;
//     double average_precision_thresholded=0;
//     double average_overlap_thresholded =0;
//
//     int totalVideos=video_gt_images.size();
//     int i=0;
//
//
//     for (i=0; i<totalVideos; i++) {
//
//       //for (int i=0; i<video_gt_images.size(); i++) {
//         Struck tracker=Struck::getTracker();
//         tracker.display=0;
//         int n=MIN(frames,video_gt_images[i].second.size());
//         EvaluationRun r= tracker.applyTrackerOnVideoWithinRange(dataset, rootFolder,saveFolder, i, 0, n);
//
//
//         average_overlap=average_overlap+r.overlap_area;
//         average_overlap_thresholded=average_overlap_thresholded+r.overlap_half;
//
//         average_precision=average_precision+r.precision_area;
//         average_precision_thresholded=average_precision_thresholded+r.precision_20;
//     }
//
//     std::time_t t2 = std::time(0);
//     //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
//
//     std::cout<<"Overlap  : "<<average_overlap_thresholded/totalVideos<<" / "<<average_overlap/totalVideos<<std::endl;
//     std::cout<<"Precision: "<<average_precision_thresholded/totalVideos<<" / "<<average_precision/(totalVideos*50)<<" / "<<average_precision/totalVideos<<std::endl;
//     std::cout<<"Time with threads: "<<(t2-t1)<<std::endl;
//
//
// }



int main(int argc, const char * argv[]) {

    // DataSetWu2013* wu2013=new DataSetWu2013;
    //
    // //DatasetALOV300* alov300=new DatasetALOV300;
    // //
    // //DatasetVOT2014* vot2014=new DatasetVOT2014;
    //
    //
    //
    //
    // std::vector<std::pair<std::string, std::vector<std::string>>>   wuPrepared=wu2013->prepareDataset(wu2013RootFolder);
    // Struck tracker=Struck::getTracker();
    //
    //
    // int frames=50;
    //
    //
    // std::cout<<wu2013RootFolder<<std::endl;
    // std::cout<<wu2013SaveFolder<<std::endl;
    //
    // applyTrackerOnDatasetParralel(wu2013, wu2013RootFolder, wu2013SaveFolder,frames);

    //applyTrackerOnDatasetSequential(wu2013, wu2013RootFolder, wu2013SaveFolder,frames);


    return 0;
}
