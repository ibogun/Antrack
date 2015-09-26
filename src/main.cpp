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

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "Features/HoG.h"

#include "Tracker/LocationSampler.h"
#include "Tracker/OLaRank_old.h"
#include "Tracker/Struck.h"
#include "Tracker/ObjStruck.h"
#include "Tracker/ObjDetectorStruck.h"
#include "Tracker/FilterBadBoxesStruck.h"

#include "Datasets/DataSetWu2013.h"
#include "Datasets/DatasetALOV300.h"
#include "Datasets/DatasetVOT2014.h"
#include "Datasets/DatasetVOT2015.h"
#include "Datasets/EvaluationRun.h"
#include "Datasets/ExperimentTemporalRobustness.h"
#include "Datasets/ExperimentSpatialRobustness.h"
#include "Datasets/ExperimentRunner.h"
#include "Datasets/AllExperimentsRunner.h"

#include "Superpixels/SuperPixels.h"

#include <pthread.h>
#include <thread>

#include <fstream>

#ifdef _WIN32
// define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
// define something for Windows (64-bit only)
#endif
#elif __APPLE__

#include "TargetConditionals.h"

#define NUM_THREADS 16

#define wu2013RootFolder "/Users/Ivan/Files/Data/wu2013/"
#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/Users/Ivan/Files/Data/vot2014/"
#define vot2015RootFolder "/Users/Ivan/Files/Data/vot2015/"

#define wu2013SaveFolder "/Users/Ivan/Files/Results/Tracking/wu2013"
#define alovSaveFolder "/Users/Ivan/Files/Results/Tracking/alov300"
#define vot2014SaveFolder "/Users/Ivan/Files/Results/Tracking/vot2014"
#define vot2015SaveFolder "/Users/Ivan/Files/Results/Tracking/vot2015"

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
#define wu2013RootFolder \
  "/udrive/student/ibogun2010/Research/Data/Tracking_benchmark/"

#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"
#define vot2015RootFolder "/Users/Ivan/Files/Data/vot2015/"  // incorrect
// OPTLEX machine
// #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"

#define wu2013SaveFolder "/udrive/student/ibogun2010/Research/Results/wu2013/"
#define alovSaveFolder "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder "/media/drive/UbuntuFiles/Results/vot2014"
#define vot2015SaveFolder "/Users/Ivan/Files/Results/Tracking/vot2015"
#elif __unix  // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif

DEFINE_double(lambda, 0.1, "Lambda in ObjDetectorStruck()");
DEFINE_double(straddling_threshold, 2, "Straddeling threshold.");
DEFINE_int32(video_index, -1, "Video to use for tracking.");
DEFINE_string(video_name, "", "Video name to use.");
DEFINE_int32(frame_from, 0, "Frame from");
DEFINE_int32(frame_to, 5000, "Frame to");

void runTrackerOnDatasetPart(
    vector<pair<string, vector<string>>> &video_gt_images, Dataset *dataset,
    int from, int to, std::string saveFolder, bool saveResults, int nFrames) {
  std::time_t t1 = std::time(0);

  int frameNumber = 0;
  // paralelize this loop
  for (int videoNumber = from; videoNumber < to; videoNumber++) {
    Struck tracker = Struck::getTracker();
    tracker.display = 0;
    pair<string, vector<string>> gt_images = video_gt_images[videoNumber];

    std::cout << gt_images.first << std::endl;

    vector<cv::Rect> groundTruth = dataset->readGroundTruth(gt_images.first);

    frameNumber += gt_images.second.size();
    cv::Mat image = cv::imread(gt_images.second[0]);

    tracker.initialize(image, groundTruth[0]);

    nFrames = MIN(nFrames, gt_images.second.size());

    for (int i = 1; i < nFrames; i++) {
      cv::Mat image = cv::imread(gt_images.second[i]);

      tracker.track(image);
    }

    if (saveResults) {
      std::string saveFileName =
          saveFolder + "/" + dataset->videos[videoNumber] + ".dat";

      tracker.saveResults(saveFileName);
    }

    EvaluationRun r;

    r.evaluate(groundTruth, tracker.boundingBoxes);

    std::cout << dataset->videos[videoNumber] << std::endl;
    std::cout << r << std::endl;

    // tracker.reset();
  }

  std::time_t t2 = std::time(0);
  std::cout << "Frames per second: " << frameNumber / (1.0 * (t2 - t1))
            << std::endl;
  // std::cout<<"No threads: "<<(t2-t1)<<std::endl;
}

vector<EvaluationRun> applyTrackerOnDataset(Dataset *dataset,
                                            std::string rootFolder,
                                            std::string saveFolder,
                                            int frames = 10) {
  vector<EvaluationRun> results;

  vector<pair<string, vector<string>>> video_gt_images =
      dataset->prepareDataset(rootFolder);

  std::time_t t1 = std::time(0);

#pragma omp parallel for
  for (int i = 0; i < video_gt_images.size(); i++) {
    std::string feature = "hist";
    std::string kernel = "int";

    Struck tracker =
        Struck::getTracker(true, true, true, true, false, kernel, feature);

    tracker.display = 0;
    int n = MIN(frames, video_gt_images[i].second.size());
    EvaluationRun r = tracker.applyTrackerOnVideoWithinRange(
        dataset, rootFolder, saveFolder, i, 0, n);

    results.push_back(r);
  }

  std::time_t t2 = std::time(0);
  // std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
  std::cout << "Time with threads: " << (t2 - t1) << std::endl;

  return results;
}

void applyTrackerOnDataset(Dataset *dataset, std::string rootFolder,
                           std::string saveFolder, bool saveResults,
                           int n_threads, int nFrames = 5000) {
  using namespace std;

  vector<pair<string, vector<string>>> video_gt_images =
      dataset->prepareDataset(rootFolder);

  std::time_t t1 = std::time(0);

  std::vector<std::thread> th;

  arma::rowvec bounds = arma::linspace<rowvec>(
      0, video_gt_images.size(), MIN(n_threads, video_gt_images.size()));

  bounds = arma::round(bounds);

  for (int i = 0; i < n_threads; i++) {
    th.push_back(std::thread(runTrackerOnDatasetPart, std::ref(video_gt_images),
                             std::ref(dataset), std::ref(bounds[i]),
                             std::ref(bounds[i + 1]), std::ref(saveFolder),
                             std::ref(saveResults), std::ref(nFrames)));
  }

  for (auto &t : th) {
    t.join();
  }

  std::time_t t2 = std::time(0);
  // std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
  std::cout << "Time with threads: " << (t2 - t1) << std::endl;

  std::ofstream out(saveFolder + "/" + "tracker_info.txt");
  out << Struck::getTracker();
  out.close();
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // DatasetALOV300* alov300=new DatasetALOV300;
  DataSetWu2013 *wu2013 = new DataSetWu2013;
  DatasetVOT2014 *vot2014 = new DatasetVOT2014;
  DatasetVOT2015 *vot2015 = new DatasetVOT2015;

  wu2013->setRootFolder(wu2013RootFolder);

  Dataset *dataset = wu2013;
  // dataset->setRootFolder(vot2015RootFolder);

  // std::vector<std::pair<std::string, std::vector<std::string>>> votPrepared=
  //    dataset->prepareDataset(vot2015RootFolder);
  std::vector<std::pair<std::string, std::vector<std::string>>> videos =
      wu2013->prepareDataset(wu2013RootFolder);

  std::string feature = "hogANDhist";
  std::string kernel = "int";

  bool pretraining = false;
  bool filter = true;
  bool straddling = false;
  bool edgeness = false;
  bool spatialPrior = false;
  std::string note = " Object Struck tracker";

  ObjDetectorStruck tracker =
      ObjDetectorStruck::getTracker(pretraining, filter, edgeness, straddling,
                                    spatialPrior, kernel, feature, note);

  tracker.setLambda(FLAGS_lambda);
  tracker.setMinStraddeling(FLAGS_straddling_threshold);
  int frames = 10;

  std::string vidName = "basketball";

  if (FLAGS_video_name != "") {
    vidName = FLAGS_video_name;
  }

  int vidIndex = dataset->vidToIndex.at(vidName);
  if (FLAGS_video_index != -1) vidIndex = FLAGS_video_index;

  // tracker.display=0;

  tracker.display = 0;

  vector<pair<string, vector<string>>> video_gt_images =
      dataset->prepareDataset(wu2013RootFolder);

  std::string save_base = wu2013SaveFolder;
  std::string dirName = save_base + "/lambda_" + std::to_string(FLAGS_lambda);
  // AllExperimentsRunner::createDirectory(dirName);

  pair<string, vector<string>> gt_images = video_gt_images[vidIndex];

  vector<cv::Rect> groundTruth = dataset->readGroundTruth(gt_images.first);

  int display = 0;

  double b = 10;

  int startingFrame = FLAGS_frame_from;
  int endingFrame = MIN(FLAGS_frame_to, groundTruth.size());

  cv::Mat image = cv::imread(gt_images.second[startingFrame]);

  std::cout << " Rect: " << groundTruth[startingFrame] << std::endl;
  std::cout << gt_images.second[startingFrame] << std::endl;
  cv::Rect rect = groundTruth[startingFrame];
  if (rect.x + rect.width >= image.cols) {
    rect.width = image.cols - rect.x - 1;
  }

  if (rect.y + rect.height >= image.rows) {
    rect.height = image.rows - rect.y - 1;
  }

  std::cout << rect << std::endl;
  std::cout << image.rows << " " << image.cols << std::endl;

  tracker.initialize(image, rect);

  for (int i = startingFrame; i < endingFrame; i++) {
    tracker.track(gt_images.second[i]);
    std::cout << "Frame #" << i - startingFrame << " out of "
              << endingFrame - startingFrame
              << tracker.getBoundingBoxes()[i - startingFrame] << std::endl;
  }

  std::string tracker_save_file = dirName + "/" + dataset->videos[vidIndex];
  // tracker.saveResults(tracker_save_file);

  delete vot2014;
  delete vot2015;
  delete wu2013;
  return 0;
}
