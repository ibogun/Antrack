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

using namespace std;
using namespace cv;


int main(int argc, const char * argv[]) {
    
    
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
    HistogramFeatures* features=new HistogramFeatures(3,16);
    // RBFKe
    //IntersectionKernel_fast* kernel=new IntersectionKernel_fast;
    ApproximateKernel* kernel= new ApproximateKernel(30);
    //IntersectionKernel* kernel=new IntersectionKernel;
    //Haar* features=new Haar(2);

    int verbose = 0;
    int display = 1;
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
    
    std::string rootFolder="/Users/Ivan/Files/Data/Tracking_benchmark/";
    std::string alovRootFolder="/Users/Ivan/Files/Data/Tracking_alov300/";
    
    std::string vot2014RootFolder="/Users/Ivan/Files/Data/vot2014/";
    DataSetWu2013* datasetWu2013=new DataSetWu2013;

    DatasetALOV300* datasetAlov=new DatasetALOV300;
    
    DatasetVOT2014* vot2014=new DatasetVOT2014;
  
    std::vector<std::pair<std::string,std::vector<std::string>>> set=vot2014->prepareDataset(vot2014RootFolder);
    using namespace std;
    using namespace cv;
    
    
    vot2014->showVideo(vot2014RootFolder,17);

    

    //tracker.applyTrackerOnDataset(datasetWu2013, rootFolder,6);
    //tracker.applyTrackerOnDataset(datasetAlov, alovRootFolder,6);

    //tracker.applyTrackerOnVideoWithinRange(datasetWu2013, rootFolder, 0, 0, 250);
    //tracker.videoCapture();
    
    return 0;
}
