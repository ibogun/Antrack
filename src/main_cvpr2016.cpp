//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <iostream>
#include <pthread.h>
#include <thread>
#include <sstream>
#include <string>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <utility>
#include <iterator>

#include "Superpixels/SuperPixels.h"


#include "Tracker/Struck.h"

#include "Datasets/AllDatasets.h"


#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>


#ifdef _WIN32
// define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
// define something for Windows (64-bit only)
#endif
#elif __APPLE__

#include "TargetConditionals.h"

#define NUM_THREADS         16

#define wu2013RootFolder    "/Users/Ivan/Files/Data/wu2013/"
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
#define wu2013RootFolder "/udrive/student/ibogun2010/Research/Data/wu2013/"

#define alovRootFolder      "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder   "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"

// OPTLEX machine
// #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"

#define wu2013SaveFolder  "/udrive/student/ibogun2010/Research/Results/wu2013/"
#define alovSaveFolder      "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder    "/media/drive/UbuntuFiles/Results/vot2014"
#elif __unix  // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif


namespace po = boost::program_options;
using std::string;
using boost::uuids::random_generator;
using boost::uuids::uuid;
using boost::lexical_cast;



DEFINE_string(datasetSaveLocation, wu2013SaveFolder,
              "Location where temporary files will be created.");
DEFINE_string(feature, "hogANDhist", "Feature (e.g. raw, hist, haar, hog).");
DEFINE_string(kernel, "int", "Kernel to use (e.g. linear, gauss, int).");
DEFINE_int32(nThreads, 1, "Number of threads to use.");
DEFINE_int32(filter, 1, "Use Robust Kalman filter on not.");
DEFINE_int32(updateEveryNframes, 3,
             "Update tracker every N frames.");
DEFINE_int32(b, 10, "Robust constant.");
DEFINE_int32(P, 10, "P in Robust Kalman filter.");
DEFINE_int32(Q, 13, "Q in Robust Kalman filter.");
DEFINE_int32(R, 13, "R in Robust Kalman filter.");
DEFINE_string(wu2013_Root_Folder, wu2013RootFolder,
              "Root folder for the wu2013 dataset.");
DEFINE_double(lambda_s, 0.3, "Straddling lambda in ObjDetectorTracker().");
DEFINE_double(lambda_e, 0.3, "Edge density lambda in ObjDetectorTracker().");
DEFINE_double(inner, 0.9, "Inner bounding box for objectness.");
DEFINE_double(straddeling_threshold, 1.5,
              "Straddeling threshold.");
DEFINE_int32(display, 0, "Display.");
DEFINE_string(prefix, "rob_", "Prefix for the tracking results.");
DEFINE_int32(experiment_type, 0,
             "Type of experiment. The smaller number - less tracking runs.");
DEFINE_int32(tracker_type, 0,
              "Type of the tracker (RobStruck - 0, ObjDet - 1, FilterBad - 2)");

DEFINE_double(topK, 50, "Top K objectness boxes in FilterBadStruck tracker.");





string make_uuid() {
    return lexical_cast<string>((random_generator()) ());
}


void experiment(int n_threads, std::string datasetTempSaveLocation,
                std::string prefix,
                std::string kernel, std::string feature, bool useFilter,
                int updateEveryNthFrames,
                double b, int P, int Q, int R,
                std::string wu2013_dataset_folder,
                int display, int experiment_type,
                const std::unordered_map<std::string, double>& map
    ) {
    // Now, run everything
    bool useEdgeDensity = false;
    bool useStraddling = false;
    bool scalePrior = false;
    std::string trackerID = "";
    DatasetWu2013 *wu2013 = new DatasetWu2013;

    std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared =
        wu2013->prepareDataset(wu2013_dataset_folder);

    bool pretraining = false;

    std::stringstream s1;
    s1 << prefix;
    trackerID = s1.str();
    // std::string folderName = make_uuid();

    std::cout << trackerID << std::endl;

    // remote location
    std::string datasetSaveLocation = datasetTempSaveLocation;

    // std::string datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
    std::string fullFolderName =
            datasetSaveLocation + "/" + trackerID;

    std::string trackerName = trackerID;

    std::cout << "Tracker name entered is: " << trackerName
    << std::endl;

    if (!(boost::filesystem::exists(fullFolderName))) {
        AllExperimentsRunner::createDirectory(fullFolderName);
    }


    std::cout<< "Flags given: "<< std::endl;
    std::cout<< "=================================================" <<std::endl;
    std::cout<< "datasetSaveLocation: " <<
        FLAGS_datasetSaveLocation <<std::endl;
    std::cout<< "feature: " << FLAGS_feature <<std::endl;
    std::cout<< "kernel: " << FLAGS_kernel <<std::endl;
    std::cout<< "nThreads: " << FLAGS_nThreads <<std::endl;
    std::cout<< "filter: " << FLAGS_filter <<std::endl;
    std::cout<< "updateEveryNframes: " << FLAGS_updateEveryNframes <<std::endl;
    std::cout<< "b: " << FLAGS_b <<std::endl;
    std::cout<< "P: " << FLAGS_P <<std::endl;
    std::cout<< "Q: " << FLAGS_Q <<std::endl;
    std::cout<< "R: " << FLAGS_R <<std::endl;
    std::cout<< "wu2013_Root_Folder: " << FLAGS_wu2013_Root_Folder <<std::endl;
    std::cout<< "updateEveryNframes: " << FLAGS_updateEveryNframes <<std::endl;
    std::cout<< "lambda_s: " << FLAGS_lambda_s <<std::endl;
    std::cout<< "lambda_e: " << FLAGS_lambda_e <<std::endl;
    std::cout<< "inner: " << FLAGS_inner <<std::endl;
    std::cout<< "straddeling_threshold: "
             << FLAGS_straddeling_threshold <<std::endl;
    std::cout<< "display: " << FLAGS_display <<std::endl;
    std::cout<< "prefix: " << FLAGS_prefix <<std::endl;
    std::cout<< "experiment_type: " << FLAGS_experiment_type <<std::endl;
    std::cout<< "tracker_type: " << FLAGS_tracker_type <<std::endl;
    std::cout<< "topK: " << FLAGS_topK <<std::endl;
    std::cout<< "===============================================" <<std::endl;

    wu2013->setRootFolder(wu2013_dataset_folder);
    AllExperimentsRunner run(wu2013);

    if (experiment_type == 0) {
        run.runSmall(fullFolderName, n_threads, true, pretraining, useFilter,
                     useEdgeDensity, useStraddling,
                     scalePrior,
                     kernel,
                     feature, updateEveryNthFrames, b, P, R, Q,
                     map, display);
    }

    if (experiment_type == 1) {
        run.run(fullFolderName, n_threads, true, pretraining, useFilter,
                useEdgeDensity, useStraddling,
                scalePrior,
                kernel,
                feature, updateEveryNthFrames, b, P, R, Q,
                map, display);
    }

    delete wu2013;
}


int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    try {
        std::string datasetSaveLocation = FLAGS_datasetSaveLocation;
        int nThreads = FLAGS_nThreads;
        int updateEveryNthFrames = FLAGS_updateEveryNframes;
        double b = FLAGS_b;
        int P = FLAGS_P;
        int Q = FLAGS_Q;
        int R = FLAGS_R;
        double straddeling_threshold = FLAGS_straddeling_threshold;
        bool filter = false;

        if (FLAGS_filter == 1) {
            filter = true;
        }

        std::string feature = FLAGS_feature;
        std::string kernel = FLAGS_kernel;
        std::string prefix = FLAGS_prefix;

        std::string wu2013_root_folder = FLAGS_wu2013_Root_Folder;
        int display = FLAGS_display;
        int experiment_type = FLAGS_experiment_type;

         std::unordered_map<std::string, double> map;

         map.insert(std::make_pair("lambda_straddling", FLAGS_lambda_s));
         map.insert(std::make_pair("lambda_edgeness", FLAGS_lambda_e));
         map.insert(std::make_pair("inner", FLAGS_inner));
         map.insert(std::make_pair("straddling_threshold",
                                   straddeling_threshold));

         map.insert(std::make_pair("topK", FLAGS_topK));

         map.insert(std::make_pair("tracker_type", FLAGS_tracker_type));

         experiment(nThreads, datasetSaveLocation, prefix,
                    kernel, feature, filter, updateEveryNthFrames, b, P, Q, R,
                    wu2013_root_folder, display,
                    experiment_type, map);
    }
    catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;
}
