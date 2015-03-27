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

#include <sstream>
#include <string>


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

#include <boost/program_options.hpp>
namespace po = boost::program_options;

void
runTrackerOnDatasetPart(vector<pair<string, vector<string>>> &video_gt_images,
                        Dataset *dataset, int from, int to,
                        std::string saveFolder, bool saveResults, int nFrames,
                        bool pretraining, bool useFilter, bool useEdgeDensity,
                        bool useStraddling, bool scalePrior ) {

    std::time_t t1 = std::time(0);

    int frameNumber = 0;
    // parallelize this loop
    for (int videoNumber=from; videoNumber<to; videoNumber++) {
        Struck tracker=Struck::getTracker(pretraining,useFilter,useEdgeDensity,useStraddling,scalePrior);
         tracker.display=0;
        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];

        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);

        frameNumber+=gt_images.second.size();
        cv::Mat image=cv::imread(gt_images.second[0]);


        tracker.initialize(image, groundTruth[0]);


        nFrames=MIN(nFrames,gt_images.second.size());
        //std::cout<<"Number of frames: "<<nFrames<<std::endl;

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


vector<EvaluationRun> applyTrackerOnDataset(Dataset* dataset, std::string rootFolder,std::string saveFolder,int frames=10){

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

void applyTrackerOnDataset(Dataset *dataset, std::string rootFolder,
                           std::string saveFolder, bool saveResults,
                           int n_threads, bool pretraining, bool useFilter,
                           bool useEdgeDensity, bool useStraddling,
                           bool scalePrior, int nFrames = 5000) {

    using namespace std;

    vector<pair<string, vector<string>>> video_gt_images=dataset->prepareDataset(rootFolder);

    std::time_t t1 = std::time(0);


    std::vector<std::thread> th;

    arma::rowvec bounds=arma::linspace<rowvec>(0, video_gt_images.size(),MIN(n_threads,video_gt_images.size()));

    bounds=arma::round(bounds);

    for (int i=0; i<n_threads; i++) {
      th.push_back(std::thread(
          runTrackerOnDatasetPart, std::ref(video_gt_images), std::ref(dataset),
          std::ref(bounds[i]), std::ref(bounds[i + 1]), std::ref(saveFolder),
          std::ref(saveResults), std::ref(nFrames), std::ref(pretraining),
          std::ref(useFilter), std::ref(useEdgeDensity),
          std::ref(useStraddling), std::ref(scalePrior)));
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

//
// int main(int argc, const char * argv[]) {
//
//     DataSetWu2013* wu2013=new DataSetWu2013;
//
//     std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared=wu2013->prepareDataset(wu2013RootFolder);
//
//     int frames = 50000;
//
//     int n_threads=50;
//
//     bool pretraining;
//     bool useFilter;
//     bool useEdgeDensity;
//     bool useStraddling;
//     bool scalePrior;
//
//     // applyTrackerOnDataset(wu2013, wu2013RootFolder,
//     // wu2013SaveFolder,true,n_threads,frames);
//
//     return 0;
// }

#include <boost/program_options.hpp>


#include <iostream>
#include <iterator>

#include <string>
using std::string;

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <boost/uuid/uuid.hpp>
using boost::uuids::uuid;

#include <boost/uuid/uuid_generators.hpp>
using boost::uuids::random_generator;

#include <boost/uuid/uuid_io.hpp>

string make_uuid()
{
    return lexical_cast<string>((random_generator())());
}

int main(int ac, char* av[])
{
    using namespace std;
    namespace po = boost::program_options;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")(
            "TrackerID", po::value<std::string>(),
            "give a name to the tracker run")(
            "Pretraining", po::value<double>(), "set flag, e.g. 0 or 1")(
            "useFilter", po::value<double>(), "set flag, e.g. 0 or 1")(
            "useEdgeDensity", po::value<double>(), "set flag, e.g. 0 or 1")(
            "useStraddling", po::value<double>(), "set flag, e.g. 0 or 1")(
            "scalePrior", po::value<double>(), "set flag, e.g. 0 or 1");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        std::string trackerID = "";

        if(vm.count("TrackerID")){
          cout << "TrackerID is " << vm["TrackerID"].as<std::string>() << ".\n";
          trackerID = vm["TrackerID"].as<std::string>();
        }



        bool pretraining,useFilter,useEdgeDensity,useStraddling,scalePrior;

        if (vm.count("Pretraining")) {
          cout << "Pretraining level was set to "
               << vm["Pretraining"].as<double>() << ".\n";

          if (vm["Pretraining"].as<double>() != 0) {
                   pretraining=true;
               }else{
                 pretraining = false;
               }
        } else {
          cout << "Pretraining flag was not set.\n";
        }

        if (vm.count("useFilter")) {
          cout << "useFilter level was set to "
               << vm["useFilter"].as<double>() << ".\n";

          if (vm["useFilter"].as<double>() != 0) {
              useFilter=true;
               }else{
                 useFilter = false;
               }
        } else {
            cout << "Filter flag was not set.\n";
        }

        if (vm.count("useEdgeDensity")) {
          cout << "useEdgeDensity level was set to "
               << vm["useEdgeDensity"].as<double>() << ".\n";

          if (vm["useEdgeDensity"].as<double>() != 0) {
              useEdgeDensity=true;
               }else{
                   useEdgeDensity = false;
               }
        } else {
          cout << "Edge density flag was not set.\n";
        }

        if (vm.count("useStraddling")) {
          cout << "useStraddling level was set to "
               << vm["useStraddling"].as<double>() << ".\n";

          if (vm["useStraddling"].as<double>() != 0) {
              useStraddling=true;
               }else{
                 useStraddling = false;
               }
        } else {
          cout << "Straddling flag was not set.\n";
        }

        if (vm.count("scalePrior")) {
          cout << "scalePrior level was set to "
               << vm["scalePrior"].as<double>() << ".\n";

          if (vm["scalePrior"].as<double>() != 0) {
              scalePrior=true;
               }else{
                   scalePrior = false;
               }
        } else {
          cout << "Edge density flag was not set.\n";
        }

        // Now, run everything
            DataSetWu2013* wu2013=new DataSetWu2013;

            std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared=wu2013->prepareDataset(wu2013RootFolder);

            int frames = 50000;

            int n_threads = 50;

            std::string folderName=make_uuid();

            std::cout << folderName << std::endl;

            std::string datasetSaveLocation="/udrive/student/ibogun2010/Research/Results";
            std::string fullFolderName =
                datasetSaveLocation + "/" + folderName + "/";

            std::cout << "Enter tracker identifier: "<<std::endl;

            std::string trackerName=trackerID;

            std::cout << "Tracker name entered is: " << trackerName
                      << std::endl;

            std::string createFolderCommand = "mkdir " + fullFolderName;
            system(createFolderCommand.c_str());
            applyTrackerOnDataset(wu2013, wu2013RootFolder, fullFolderName,
                                  true, n_threads, pretraining, useFilter,
                                  useEdgeDensity, useStraddling, scalePrior,
                                  frames);


            std::stringstream ss;

            ss << "python "
               << "/udrive/student/ibogun2010/Research/Code/Antrack/python/"
               << "Evaluation/generatePythonFilePickle.py"
               << " " << fullFolderName << " "
               << "wu2013"
               << " " << trackerName << " "
               << "/udrive/student/ibogun2010/Research/"
               << "Code/Antrack/python/Evaluation/Runs/";

            std::string createPickleCommand = ss.str();

            system(createPickleCommand.c_str());
            std::cout << "Pickel was created" << std::endl;

            // now delete the folder

            std::string deletecommand="rm -rf "+fullFolderName;

            system(deletecommand.c_str());

    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;
}
