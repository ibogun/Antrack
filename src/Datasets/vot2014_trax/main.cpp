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

#include "../../Kernels/RBFKernel.h"
#include "../../Kernels/IntersectionKernel.h"
#include "../../Kernels/IntersectionKernel_fast.h"
#include "../../Kernels/ApproximateKernel.h"
#include "../../Kernels/LinearKernel.h"

#include "../../Features/RawFeatures.h"
#include "../../Features/Haar.h"
#include "../../Features/Histogram.h"
#include "../../Features/HoG.h"

#include "../../Tracker/LocationSampler.h"
#include "../../Tracker/OLaRank_old.h"
#include "../../Tracker/Struck.h"

#include "../../Datasets/DataSetWu2013.h"
#include "../../Datasets/DatasetALOV300.h"
#include "../../Datasets/DatasetVOT2014.h"

#include <pthread.h>
#include <thread>

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

//#ifdef TRAX
#include "trax.h"
//#else
//#include "vot.hpp"
//#endif


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
#define NUM_THREADS         8

#define wu2013RootFolder    "/Users/Ivan/Files/Data/Tracking_benchmark/"
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
    int display = 0;
    int m       = features->calculateFeatureDimension();
    
    OLaRank_old* olarank=new OLaRank_old(kernel);
    olarank->setParameters(p, B,m,verbose);
    
    int r_search = 30;
    int r_update = 60;
    
    bool useFilter=false;
    bool useObjectness=false;
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

//#ifdef TRAX
int main( int argc, char** argv)
{

    Struck tracker=getTracker();
    trax_image* img = NULL;
    trax_region* reg = NULL;
    trax_region* mem = NULL;

    trax_handle* trax;
    trax_configuration config;
    config.format_region = TRAX_REGION_RECTANGLE;
    config.format_image = TRAX_IMAGE_PATH;

    trax = trax_server_setup_standard(config, NULL);

    bool run = true;

    while(run)
    {

        int tr = trax_server_wait(trax, &img, &reg, NULL);

        if (tr == TRAX_INITIALIZE) {

            cv::Rect rect;
            float x, y, width, height;
            trax_region_get_rectangle(reg, &x, &y, &width, &height);
            rect.x = round(x); rect.y = round(y); rect.width = round(x + width) - rect.x; rect.height = round(y + height) - rect.y;

            cv::Mat image = cv::imread(trax_image_get_path(img));

            tracker.initialize(image, rect);

            trax_server_reply(trax, reg, NULL);

        } else if (tr == TRAX_FRAME) {

            cv::Mat image = cv::imread(trax_image_get_path(img));

            cv::Rect rect = tracker.track(image);

            trax_region* result = trax_region_create_rectangle(rect.x, rect.y, rect.width, rect.height);

            trax_server_reply(trax, result, NULL);

            trax_region_release(&result);

        } else {

            run = false;

        }

        if (img) trax_image_release(&img);
        if (reg) trax_region_release(&reg);

    }

    if (mem) trax_region_release(&mem);

    trax_cleanup(&trax);

    return 0;

}
