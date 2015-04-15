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
#include "Datasets/ExperimentTemporalRobustness.h"
#include "Datasets/ExperimentSpatialRobustness.h"
#include "Datasets/ExperimentRunner.h"
#include "Datasets/AllExperimentsRunner.h"

#include "Superpixels/SuperPixels.h"

#include <pthread.h>
#include <thread>

#include <fstream>
#include <boost/filesystem.hpp>
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

string make_uuid() {
    return lexical_cast<string>((random_generator()) ());
}


void experimentForFilter(int n_threads, std::string datasetTempSaveLocation,std::string prefix,
                         std::string kernel, std::string feature) {
    // Now, run everything


    double b=10;
    int P=3;
    int R=5;
    int Q=5;
    bool useEdgeDensity = false;
    bool useStraddling = false;
    bool pretraining = false;
    bool scalePrior = false;

    std::vector<std::string> kernels;
    std::vector<std::string> features;
    std::vector<bool> filter_flags;


    filter_flags.push_back(false);
    filter_flags.push_back(true);

    std::string trackerID = "";


    for (int k = 0; k < filter_flags.size(); ++k) {


        DataSetWu2013 *wu2013 = new DataSetWu2013;

        std::vector<std::pair<std::string, std::vector<std::string> > > wuPrepared = wu2013->prepareDataset(
                wu2013RootFolder);

        bool useFilter = filter_flags[k];

        std::stringstream s1;

        s1 << prefix << feature << "_" << kernel;


        s1 << "_f";

        if (useFilter) {
            s1 << "1";
        } else {
            s1 << "0";
        }
        trackerID = s1.str();
        //std::string folderName = make_uuid();

        std::cout << trackerID << std::endl;

        // remote location
        std::string datasetSaveLocation = datasetTempSaveLocation;

        //std::string datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
        std::string fullFolderName =
                datasetSaveLocation + "/" + trackerID;

        std::cout << "Enter tracker identifier: " << std::endl;

        std::string trackerName = trackerID;

        std::cout << "Tracker name entered is: " << trackerName
        << std::endl;

        if (!(boost::filesystem::exists(fullFolderName))) {

            AllExperimentsRunner::createDirectory(fullFolderName);
        }


        wu2013->setRootFolder(wu2013RootFolder);
        AllExperimentsRunner run(wu2013);


        run.run(fullFolderName, n_threads, true, pretraining, useFilter, useEdgeDensity, useStraddling,
                scalePrior,
                kernel,
                feature, b, P, R, Q);


        delete wu2013;

    }


}

void experimentSensitivityOtherParams(int n_threads, std::string datasetTempSaveLocation,std::string prefix,
                                      std::string kernel, std::string feature, double b, int P, int Q, int R) {
    // Now, run everything

    bool useEdgeDensity = false;
    bool useStraddling = false;

    bool scalePrior = false;


    std::string trackerID = "";


    DataSetWu2013 *wu2013 = new DataSetWu2013;

    std::vector<std::pair<std::string, std::vector<std::string> > > wuPrepared = wu2013->prepareDataset(
            wu2013RootFolder);

    bool pretraining = false;
    bool useFilter = true;

    std::stringstream s1;

    s1 << prefix << feature << "_" << kernel;


    s1 << "_f";

    if (useFilter) {
        s1 << "1";
    } else {
        s1 << "0";
    }
    trackerID = s1.str();
    //std::string folderName = make_uuid();

    std::cout << trackerID << std::endl;

    // remote location
    std::string datasetSaveLocation = datasetTempSaveLocation;

    //std::string datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
    std::string fullFolderName =
            datasetSaveLocation + "/" + trackerID;

    std::string trackerName = trackerID;

    std::cout << "Tracker name entered is: " << trackerName
    << std::endl;

    if (!(boost::filesystem::exists(fullFolderName))) {

        AllExperimentsRunner::createDirectory(fullFolderName);
    }


    wu2013->setRootFolder(wu2013RootFolder);
    AllExperimentsRunner run(wu2013);


    run.run(fullFolderName, n_threads, true, pretraining, useFilter, useEdgeDensity, useStraddling,
            scalePrior,
            kernel,
            feature, b, P, R, Q);


    delete wu2013;


}

void experimentSensitivityToParameters(int n_threads, std::string datasetTempSaveLocation, std::string prefix, double b,
                                       int P, int R, int Q) {
    // Now, run everything


    bool useEdgeDensity = false;
    bool useStraddling = false;

    bool scalePrior = false;


    //std::string prefix = "b="+std::to_string(b)+"_";
    std::string trackerID = "";


    std::string kernel = "int";
    std::string feature = "hist";

    DataSetWu2013 *wu2013 = new DataSetWu2013;

    std::vector<std::pair<std::string, std::vector<std::string> > > wuPrepared = wu2013->prepareDataset(
            wu2013RootFolder);

    bool pretraining = false;
    bool useFilter = true;

    std::stringstream s1;

    s1 << prefix << feature << "_" << kernel << "_" << "pre";

    if (pretraining) {
        s1 << "1";
    } else {
        s1 << "0";
    }

    s1 << "_f";

    if (useFilter) {
        s1 << "1";
    } else {
        s1 << "0";
    }
    trackerID = s1.str();
    //std::string folderName = make_uuid();

    std::cout << trackerID << std::endl;

    // remote location
    std::string datasetSaveLocation = datasetTempSaveLocation;

    //std::string datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
    std::string fullFolderName =
            datasetSaveLocation + "/" + trackerID;

    std::cout << "Enter tracker identifier: " << std::endl;

    std::string trackerName = trackerID;

    std::cout << "Tracker name entered is: " << trackerName
    << std::endl;

    if (!(boost::filesystem::exists(fullFolderName))) {

        AllExperimentsRunner::createDirectory(fullFolderName);
    }


    wu2013->setRootFolder(wu2013RootFolder);
    AllExperimentsRunner run(wu2013);


    run.run(fullFolderName, n_threads, true, pretraining, useFilter, useEdgeDensity, useStraddling,
            scalePrior,
            kernel,
            feature, b, P, R, Q);


    delete wu2013;


}


int main(int ac, char *av[]) {
    using namespace std;
    namespace po = boost::program_options;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")(
                "tmpSaveLocation", po::value<std::string>(), "Location where temporary files will be created.")(
                "feature", po::value<std::string>(), "Feature (e.g. raw, hist, haar, hog)")(
                "kernel", po::value<std::string>(), "Kernel (e.g. linear, gauss, int)")(
                "nThreads", po::value<int>(), "Number of threads")(
                "b", po::value<double>(), " Robust constant")(
                "P", po::value<int>(), "P")(
                "Q", po::value<int>(), "Q")(
                "R", po::value<int>(), "R")(
                "prefix", po::value<std::string>(),
                "Experiment prefix (i.g. b=${b} for sensitivity to b, Q=${Q} for sensitivity to Q)");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        std::string datasetSaveLocation = "";
        std::string permamentSaveLoation = "";
        int index = 0;
        int nThreads = 1;

        double b = 10;
        int P = 3;
        int Q = 5;
        int R = 5;

        std::string feature = "raw";
        std::string kernel = "linear";
        std::string prefix = "ms";

        if (vm.count("feature")) {
            cout << "Feature is: " << vm["feature"].as<std::string>() << ".\n";
            feature = vm["feature"].as<std::string>();
        }

        if (vm.count("kernel")) {
            cout << "Kernel is: " << vm["kernel"].as<std::string>() << ".\n";
            kernel = vm["kernel"].as<std::string>();
        }

        if (vm.count("prefix")) {
            cout << "Temporary save Location is: " << vm["prefix"].as<std::string>() << ".\n";
            prefix = vm["prefix"].as<std::string>();
        }

        if (vm.count("tmpSaveLocation")) {
            cout << "Temporary save Location is: " << vm["tmpSaveLocation"].as<std::string>() << ".\n";
            datasetSaveLocation = vm["tmpSaveLocation"].as<std::string>();
        }

        if (vm.count("P")) {
            cout << "P: " << vm["P"].as<int>() << ".\n";
            P = vm["P"].as<int>();
        }

        if (vm.count("Q")) {
            cout << "Q: " << vm["Q"].as<int>() << ".\n";
            Q = vm["Q"].as<int>();
        }

        if (vm.count("R")) {
            cout << "R: " << vm["R"].as<int>() << ".\n";
            R = vm["R"].as<int>();
        }

        if (vm.count("nThreads")) {
            cout << "Number of threads is: " << vm["nThreads"].as<int>() << ".\n";
            nThreads = vm["nThreads"].as<int>();
        }

        if (vm.count("b")) {
            cout << "Robust constant is is: " << vm["b"].as<double>() << ".\n";
            b = vm["b"].as<double>();
        }


        experimentForFilter(nThreads,datasetSaveLocation,prefix,kernel,feature);
        experimentSensitivityOtherParams(nThreads,datasetSaveLocation,prefix,
                kernel,feature,b,P,Q,R);
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
