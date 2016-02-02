//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <ctime>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <pthread.h>
#include <thread>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "Tracker/AllTrackers.h"
#include "Datasets/AllDatasets.h"

#include "Superpixels/SuperPixels.h"

#ifdef _WIN32
// define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
// define something for Windows (64-bit only)
#endif
#elif __APPLE__

#include "TargetConditionals.h"

#define NUM_THREADS 16

#define wu2013RootFolder "/Users/Ivan/Files/Data/wu2013/"
#define wu2015RootFolder "/Users/Ivan/Files/Data/wu2015/"
#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/Users/Ivan/Files/Data/vot2014/"
#define vot2015RootFolder "/Users/Ivan/Files/Data/vot2015/"

#define wu2015SaveFolder "/Users/Ivan/Files/Results/Tracking/wu2015"
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
#define wu2013RootFolder "/udrive/student/ibogun2010/Research/Data/wu2013/"
#define wu2015RootFolder "/udrive/student/ibogun2010/Research/Data/wu2015/"

#define alovRootFolder "/Users/Ivan/Files/Data/Tracking_alov300/"
#define vot2014RootFolder "/media/drive/UbuntuFiles/Datasets/Tracking/vot2014"
#define vot2015RootFolder "/Users/Ivan/Files/Data/vot2015/" // incorrect
// OPTLEX machine
// #define wu2013SaveFolder    "/media/drive/UbuntuFiles/Results/wu2013"

#define wu2015SaveFolder "/udrive/student/ibogun2010/Research/Results/wu2015/"
#define wu2013SaveFolder "/udrive/student/ibogun2010/Research/Results/wu2013/"
#define alovSaveFolder "/media/drive/UbuntuFiles/Results/alov300"
#define vot2014SaveFolder "/media/drive/UbuntuFiles/Results/vot2014"
#define vot2015SaveFolder "/Users/Ivan/Files/Results/Tracking/vot2015"
#elif __unix // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif

DEFINE_int32(budget, 100, "Budget");
DEFINE_int32(display, 1, "Display settings.");
DEFINE_double(lambda_s, 0.3, "Straddling lambda in ObjDetectorTracker().");
DEFINE_double(lambda_e, 0.3, "Edge density lambda in ObjDetectorTracker().");
DEFINE_double(inner, 0.9, "Inner bounding box for objectness.");
DEFINE_double(straddeling_threshold, 1.5, "Straddeling threshold.");
DEFINE_int32(video_index, -1, "Video to use for tracking.");
DEFINE_string(video_name, "", "Video name to use.");
DEFINE_int32(frame_from, 0, "Frame from");
DEFINE_int32(frame_to, 5000, "Frame to");
DEFINE_int32(tracker_type, 1,
             "Type of the tracker (RobStruck - 0, ObjDet - 1, FilterBad - 2)");
DEFINE_double(topK, 50, "Top K objectness boxes in FilterBadStruck tracker.");
DEFINE_string(proto_file,
              "/Users/Ivan/Code/Tracking/Antrack/data/imagenet_memory.prototxt",
              "Proto file for the feature extraction using deep ConvNet");
DEFINE_string(
    conv_deep_weights,
    "/Users/Ivan/Code/Tracking/Antrack/data/bvlc_reference_caffenet.caffemodel",
    "File with the weights for the deep ConvNet");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Dataset *dataset = new DatasetWu2015;
  dataset->setRootFolder(wu2015RootFolder);

  // std::vector<std::pair<std::string, std::vector<std::string>>> votPrepared=
  //    dataset->prepareDataset(vot2015RootFolder);
  std::vector<std::pair<std::string, std::vector<std::string>>> videos =
      dataset->prepareDataset(wu2015RootFolder);

  std::string feature = "deep";
  std::string kernel = "int";

  bool pretraining = false;
  bool filter = true;
  bool straddling = false;
  bool edgeness = false;
  bool spatialPrior = false;
  std::string note = " Object Struck tracker";

  // tracker.setLambda(FLAGS_lambda);
  // tracker.setMinStraddeling(FLAGS_straddling_threshold);
  int frames = 10;

  std::string vidName = "Basketball";

  if (FLAGS_video_name != "") {
    vidName = FLAGS_video_name;
  }

  int vidIndex = dataset->vidToIndex.at(vidName);
  if (FLAGS_video_index != -1)
    vidIndex = FLAGS_video_index;

  // tracker.display=0;

  std::vector<std::pair<std::string, std::vector<std::string>>>
      video_gt_images = dataset->prepareDataset(wu2015RootFolder);

  std::string save_base = wu2015SaveFolder;
  std::string dirName = save_base + "/lambda_" + std::to_string(FLAGS_lambda_s);
  // AllExperimentsRunner::createDirectory(dirName);

  // for (vidIndex = 0; vidIndex <51 ; ++vidIndex) {

  int tracker_type = FLAGS_tracker_type;

  int useFilter = filter;
  bool useStraddling = straddling;
  bool useEdgeDensity = edgeness;
  bool scalePrior = spatialPrior;

  // ObjDetectorStruck tracker(pretraining, filter, edgeness,
  //                             straddling,
  //                             spatialPrior, kernel, feature, note);

  std::unordered_map<std::string, double> map;
  map.insert(std::make_pair("lambda_straddling", FLAGS_lambda_s));
  map.insert(std::make_pair("lambda_edgeness", FLAGS_lambda_e));
  map.insert(std::make_pair("inner", FLAGS_inner));
  map.insert(
      std::make_pair("straddling_threshold", FLAGS_straddeling_threshold));
  map.insert(std::make_pair("topK", FLAGS_topK));


  std::unordered_map<std::string, std::string> featureParamsMap;

  featureParamsMap.insert(std::make_pair("proto", FLAGS_proto_file));
  featureParamsMap.insert(std::make_pair("weights", FLAGS_conv_deep_weights));

  for (int vidIndex = 54; vidIndex < video_gt_images.size(); vidIndex++) {
    std::cout << "Vid Index: " << vidIndex  << "\n";
    pair<string, vector<string>> gt_images = video_gt_images[vidIndex];

    vector<cv::Rect> groundTruth = dataset->readGroundTruth(gt_images.first);
    Struck *tracker =
        new ObjDetectorStruck(pretraining, useFilter, useEdgeDensity,
                              useStraddling, scalePrior, kernel, feature, note);
    tracker->setParams(map);
    tracker->setFeatureParams(featureParamsMap);
    tracker->display = FLAGS_display;

    double b = 10;

    int startingFrame = FLAGS_frame_from;
    int endingFrame = MIN(FLAGS_frame_to, groundTruth.size());

    cv::Mat image = cv::imread(gt_images.second[startingFrame]);

    std::cout << " Rect: " << groundTruth[startingFrame] << std::endl;
    std::cout << gt_images.second[startingFrame] << std::endl;
    cv::Rect rect = groundTruth[startingFrame];

    std::string saveTrackingImage = "/Users/Ivan/Documents/Papers/My_papers/"
                                    "CVPR_2016_Object-aware_tracking/images/"
                                    "tracking/";
    std::string prefix = "tracking_";
    if (rect.x + rect.width >= image.cols) {
      rect.width = image.cols - rect.x - 1;
    }

    if (rect.y + rect.height >= image.rows) {
      rect.height = image.rows - rect.y - 1;
    }

    std::cout << rect << std::endl;
    std::cout << image.rows << " " << image.cols << std::endl;

    tracker->initialize(image, rect);

    for (int i = startingFrame; i < endingFrame; i++) {
      tracker->track(gt_images.second[i]);
      std::cout << "Frame #" << i - startingFrame << " out of "
                << endingFrame - startingFrame
                << tracker->getBoundingBoxes()[i - startingFrame] << std::endl;
      cv::Mat tracking_image = tracker->getObjectnessCanvas();
      // std::string savefilename= saveTrackingImage+prefix +
      // std::to_string(1000+i) + ".png";
      // std::cout<< "saving to file: " << savefilename << std::endl;
      // cv::imwrite(savefilename, tracking_image);
    }

    delete tracker;
  }
  // std::string tracker_save_file = dirName + "/" + dataset->videos[vidIndex];
  // tracker->saveResults(tracker_save_file);

  delete dataset;
  //delete tracker;
  return 0;
}
