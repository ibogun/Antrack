//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "Tracker/Struck.h"
#include "Tracker/ObjDetectorStruck.h"


DEFINE_int32(budget, 100, "Budget");
DEFINE_int32(display, 1, "Display settings.");
DEFINE_double(lambda_s, 0.3, "Straddling lambda in ObjDetectorTracker().");
DEFINE_double(lambda_e, 0.3, "Edge density lambda in ObjDetectorTracker().");
DEFINE_double(inner, 0.9, "Inner bounding box for objectness.");
DEFINE_double(straddeling_threshold, 1.5,
              "Straddeling threshold.");
DEFINE_int32(frame_from, 0, "Frame from");
DEFINE_int32(frame_to, 5000, "Frame to");
DEFINE_int32(tracker_type, 1,
             "Type of the tracker (RobStruck - 0, ObjDet - 1, FilterBad - 2)");
DEFINE_double(topK, 50, "Top K objectness boxes in FilterBadStruck tracker.");


int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);



  std::string feature = "hogANDhist";
  std::string kernel = "int";

  bool pretraining = false;
  bool filter = true;
  bool straddling = false;
  bool edgeness = false;
  bool spatialPrior = false;
  std::string note = " Object Struck tracker";

  int frames = 10;

  int tracker_type = FLAGS_tracker_type;
    pair<string, vector<string>> gt_images = video_gt_images[vidIndex];

    vector<cv::Rect> groundTruth = dataset->readGroundTruth(gt_images.first);

    int useFilter = filter;
    bool useStraddling = straddling;
    bool useEdgeDensity = edgeness;
    bool scalePrior = spatialPrior;

    Struck* tracker = new
        ObjDetectorStruck(pretraining, useFilter,
                          useEdgeDensity, useStraddling,
                          scalePrior,
                          kernel,
                          feature, note);

    //ObjDetectorStruck tracker(pretraining, filter, edgeness,
    //                             straddling,
    //                             spatialPrior, kernel, feature, note);

    std::unordered_map<std::string, double> map;
    map.insert(std::make_pair("lambda_straddling", FLAGS_lambda_s));
    map.insert(std::make_pair("lambda_edgeness", FLAGS_lambda_e));
    map.insert(std::make_pair("inner", FLAGS_inner));
    map.insert(std::make_pair("straddling_threshold",
                              FLAGS_straddeling_threshold));
    map.insert(std::make_pair("topK", FLAGS_topK));
    tracker->setParams(map);
    tracker->display = FLAGS_display;

    double b = 10;

    cv::Mat image = cv::imread(gt_images.second[startingFrame]);
    tracker->initialize(image, rect);

    for (int i = startingFrame; i < endingFrame; i++) {
        tracker->track(gt_images.second[i]);
        std::cout << "Frame #" << i - startingFrame << " out of "
                  << endingFrame - startingFrame
                  << tracker->getBoundingBoxes()[i - startingFrame] << std::endl;
        cv::Mat tracking_image = tracker->getObjectnessCanvas();
        std::string savefilename= saveTrackingImage+prefix +  std::to_string(1000+i) + ".png";
        std::cout<< "saving to file: " << savefilename << std::endl;
        cv::imwrite(savefilename, tracking_image);
    }

  std::string tracker_save_file = dirName + "/" + dataset->videos[vidIndex];
  tracker->saveResults(tracker_save_file);

  delete vot2014;
  delete vot2015;
  delete wu2013;
  return 0;
}
