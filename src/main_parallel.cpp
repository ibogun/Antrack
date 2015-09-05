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

#include <sstream>
#include <string>

#ifdef _WIN32
// define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
// define something for Windows (64-bit only)
#endif
#elif __APPLE__

#include "TargetConditionals.h"

#define NUM_THREADS 16

#define wu2013RootFolder "/Users/Ivan/Files/Data/Tracking_benchmark/"
#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/Users/Ivan/Files/Data/vot2014/"

#define wu2013SaveFolder "/Users/Ivan/Files/Results/Tracking/wu2013"
#define alovSaveFolder "/Users/Ivan/Files/Results/Tracking/alov300"
#define vot2014SaveFolder "/Users/Ivan/Files/Results/Tracking/vot2014"

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
#define NUM_THREADS 16

// OPTLEX machine
//#define wu2013RootFolder "/media/drive/UbuntuFiles/Datasets/Tracking/wu2013/"
#define wu2013RootFolder                                                       \
  "/udrive/student/ibogun2010/Research/Data/Tracking_benchmark/"

#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"

// OPTLEX machine
// #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"

#define wu2013SaveFolder "/udrive/student/ibogun2010/Research/Results/wu2013/"
#define alovSaveFolder "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder "/media/drive/UbuntuFiles/Results/vot2014"
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

string make_uuid() { return lexical_cast<string>((random_generator())()); }

void experimentForFilterAndPretraining(int n_threads) {
  // Now, run everything
  DataSetWu2013 *wu2013 = new DataSetWu2013;

  std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared =
      wu2013->prepareDataset(wu2013RootFolder);

  bool useEdgeDensity = false;
  bool useStraddling = false;

  bool scalePrior = false;

  std::vector<std::string> kernels;
  std::vector<std::string> features;
  std::vector<bool> pretraining_flags;
  std::vector<bool> filter_flags;

  pretraining_flags.push_back(false);
  pretraining_flags.push_back(true);

  filter_flags.push_back(false);
  filter_flags.push_back(true);

  kernels.push_back("int");    // hist
  kernels.push_back("linear"); // raw
  kernels.push_back("gauss");  // hog
  kernels.push_back("int");    // hog
  kernels.push_back("linear"); // hog

  features.push_back("hist");
  features.push_back("raw");
  features.push_back("hog");
  features.push_back("hog");
  features.push_back("hog");

  std::string prefix = "ms_";
  std::string trackerID = "";

  for (int i = 0; i < kernels.size(); ++i) {

    std::string kernel = kernels[i];
    std::string feature = features[i];

    for (int j = 0; j < pretraining_flags.size(); ++j) {
      for (int k = 0; k < filter_flags.size(); ++k) {

        bool pretraining = pretraining_flags[j];
        bool useFilter = filter_flags[k];

        std::stringstream s1;

        s1 << prefix << trackerID << feature << "_" << kernel << "_"
           << "pre";

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
        std::string folderName = make_uuid();

        std::cout << folderName << std::endl;

        // remote location
        std::string datasetSaveLocation =
            "/udrive/student/ibogun2010/Research/Results";

        // std::string
        // datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
        std::string fullFolderName = datasetSaveLocation + "/" + folderName;

        std::cout << "Enter tracker identifier: " << std::endl;

        std::string trackerName = trackerID;

        std::cout << "Tracker name entered is: " << trackerName << std::endl;

        std::string createFolderCommand = "mkdir " + fullFolderName;

        AllExperimentsRunner::createDirectory(fullFolderName);

        wu2013->setRootFolder(wu2013RootFolder);
        AllExperimentsRunner run(wu2013);

        run.run(fullFolderName, n_threads, true, pretraining, useFilter,
                useEdgeDensity, useStraddling, scalePrior, kernel, feature);

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

        AllExperimentsRunner::deleteDirectory(fullFolderName);
      }
    }
  }
}

int main(int ac, char *av[]) {
  using namespace std;
  namespace po = boost::program_options;
  try {
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "TrackerID", po::value<std::string>(),
        "give a name to the tracker run")("features", po::value<std::string>(),
                                          "Feature to use")(
        "kernel", po::value<std::string>(), "Kernel to use")(
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

    if (vm.count("TrackerID")) {
      cout << "TrackerID is " << vm["TrackerID"].as<std::string>() << ".\n";
      trackerID = vm["TrackerID"].as<std::string>();
    }

    std::string feature = "raw";
    if (vm.count("features")) {
      cout << "Feature is " << vm["features"].as<std::string>() << ".\n";
      feature = vm["features"].as<std::string>();
    }

    std::string kernel = "hist";
    if (vm.count("kernel")) {
      cout << "Kernel is " << vm["kernel"].as<std::string>() << ".\n";
      kernel = vm["kernel"].as<std::string>();
    }

    bool pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior;

    if (vm.count("Pretraining")) {
      cout << "Pretraining level was set to " << vm["Pretraining"].as<double>()
           << ".\n";

      if (vm["Pretraining"].as<double>() != 0) {
        pretraining = true;
      } else {
        pretraining = false;
      }
    } else {
      cout << "Pretraining flag was not set.\n";
    }

    if (vm.count("useFilter")) {
      cout << "useFilter level was set to " << vm["useFilter"].as<double>()
           << ".\n";

      if (vm["useFilter"].as<double>() != 0) {
        useFilter = true;
      } else {
        useFilter = false;
      }
    } else {
      cout << "Filter flag was not set.\n";
    }

    if (vm.count("useEdgeDensity")) {
      cout << "useEdgeDensity level was set to "
           << vm["useEdgeDensity"].as<double>() << ".\n";

      if (vm["useEdgeDensity"].as<double>() != 0) {
        useEdgeDensity = true;
      } else {
        useEdgeDensity = false;
      }
    } else {
      cout << "Edge density flag was not set.\n";
    }

    if (vm.count("useStraddling")) {
      cout << "useStraddling level was set to "
           << vm["useStraddling"].as<double>() << ".\n";

      if (vm["useStraddling"].as<double>() != 0) {
        useStraddling = true;
      } else {
        useStraddling = false;
      }
    } else {
      cout << "Straddling flag was not set.\n";
    }

    if (vm.count("scalePrior")) {
      cout << "scalePrior level was set to " << vm["scalePrior"].as<double>()
           << ".\n";

      if (vm["scalePrior"].as<double>() != 0) {
        scalePrior = true;
      } else {
        scalePrior = false;
      }
    } else {
      cout << "Edge density flag was not set.\n";
    }

    // Now, run everything
    DataSetWu2013 *wu2013 = new DataSetWu2013;

    std::vector<std::pair<std::string, std::vector<std::string>>> wuPrepared =
        wu2013->prepareDataset(wu2013RootFolder);

    int frames = 50000;

    int n_threads = 240;

    std::string folderName = make_uuid();

    std::cout << folderName << std::endl;

    // remote location
    std::string datasetSaveLocation =
        "/udrive/student/ibogun2010/Research/Results";

    // std::string datasetSaveLocation="/Users/Ivan/Code/Tracking/Antrack/tmp";
    std::string fullFolderName = datasetSaveLocation + "/" + folderName;

    std::cout << "Enter tracker identifier: " << std::endl;

    std::string trackerName = trackerID;

    std::cout << "Tracker name entered is: " << trackerName << std::endl;

    std::string createFolderCommand = "mkdir " + fullFolderName;

    AllExperimentsRunner::createDirectory(fullFolderName);

    wu2013->setRootFolder(wu2013RootFolder);
    AllExperimentsRunner run(wu2013);

    run.run(fullFolderName, n_threads, true, pretraining, useFilter,
            useEdgeDensity, useStraddling, scalePrior, kernel, feature);
    // this line has to be replaced
    //        applyTrackerOnDataset(wu2013, wu2013RootFolder, fullFolderName,
    //                              true, n_threads, pretraining, useFilter,
    //                              useEdgeDensity, useStraddling, scalePrior,
    //                              kernel, feature, frames);

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

    AllExperimentsRunner::deleteDirectory(fullFolderName);

  } catch (exception &e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    cerr << "Exception of unknown type!\n";
  }

  return 0;
}
