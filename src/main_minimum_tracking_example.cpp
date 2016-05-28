//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "Tracker/AllTrackers.h"

DEFINE_int32(tracker_type, 1,
             "Type of the tracker (RobStruck - 0, ObjStruck - 1, MBestStruck - 2)");
DEFINE_int32(budget, 100, "OLaRank's Structured SSVM Budget");
DEFINE_int32(display, 1, "Display settings (0 - no, 1 - basic, 2 - support vectors, 3 - special (only for ObjStruck)).");
DEFINE_double(lambda_s, 0.3, "Straddling lambda in ObjStruck.");
DEFINE_double(lambda_e, 0.4, "Edge density lambda in ObjStruck.");
DEFINE_double(inner, 0.9, "Inner bounding box for objectness.");
DEFINE_double(straddeling_threshold, 1.5,
              "Straddeling threshold.");

DEFINE_string(dis_feature, "hog", "Dissimilarity features to use");
DEFINE_string(dis_kernel, "linear", "Dissimilarity top features to use");

DEFINE_string(feature, "hogANDhist", "Features to use");
DEFINE_string(top_feature, "deep", "Top features to use");
DEFINE_string(top_kernel, "linear", "Top kernel to use");


// See https://gist.github.com/ibogun/092a2305bdb5de0010336ef370dfbea3
DEFINE_string(
  proto_file,
  "/Users/Ivan/Code/Tracking/DeepAntrack/data/imagenet_memory.prototxt",
  "Proto file for the feature extraction using deep ConvNet");
DEFINE_string(conv_deep_weights, "/Users/Ivan/Code/Tracking/DeepAntrack/data/"
              "bvlc_reference_caffenet.caffemodel",
              "File with the weights for the deep ConvNet");

DEFINE_double(lambda_diff, 0.1, "Lambda used in multiple hypothesis.");
DEFINE_int32(MBest, 32, "MBest M.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::string feature = FLAGS_feature;
  std::string kernel = "int";

  bool pretraining = false;
  bool filter = true;
  bool straddling = true;
  bool edgeness = true;
  bool spatialPrior = false;
  std::string note = " Object Struck tracker";

  int tracker_type = FLAGS_tracker_type;
  int useFilter = filter;
  bool useStraddling = straddling;
  bool useEdgeDensity = edgeness;
  bool scalePrior = spatialPrior;

  Struck* tracker;

  std::unordered_map<std::string, double> map;
  if (tracker_type == 1) {
    tracker = new
      ObjDetectorStruck(pretraining, useFilter,
                        useEdgeDensity, useStraddling,
                        scalePrior,
                        kernel,
                        feature, note);
  } else if (tracker_type == 2) {
    tracker = new
      MBestStruck(pretraining, useFilter,
                  useEdgeDensity, useStraddling,
                  scalePrior,
                  kernel,
                  feature, note);

    std::unordered_map<std::string, std::string> featureParamsMap;

    featureParamsMap.insert(std::make_pair("dis_features", FLAGS_dis_feature));
    featureParamsMap.insert(std::make_pair("dis_kernel", FLAGS_dis_kernel));

    featureParamsMap.insert(std::make_pair("top_features", FLAGS_top_feature));
    featureParamsMap.insert(std::make_pair("top_kernel", FLAGS_top_kernel));

    featureParamsMap.insert(std::make_pair("proto", FLAGS_proto_file));
    featureParamsMap.insert(std::make_pair("weights", FLAGS_conv_deep_weights));
    //tracker->setLambda(FLAGS_lambda_diff);
    //tracker->setM(FLAGS_MBest);
    tracker->setFeatureParams(featureParamsMap);
  } else {
    tracker = new
      Struck(pretraining, useFilter,
                        useEdgeDensity, useStraddling,
                        scalePrior,
                        kernel,
                        feature, note);
  }


    map.insert(std::make_pair("lambda_straddling", FLAGS_lambda_s));
    map.insert(std::make_pair("lambda_edgeness", FLAGS_lambda_e));
    map.insert(std::make_pair("inner", FLAGS_inner));
    map.insert(std::make_pair("straddling_threshold",
                              FLAGS_straddeling_threshold));
    tracker->setParams(map);
    tracker->display = FLAGS_display;
    cv::Rect rect(198, 214, 34, 81);

    std::string rootFolder ="../sample_data/";


    std::string fileName = rootFolder +"0001.jpg";
    cv::Mat image = cv::imread(fileName);
    tracker->initialize(image, rect);


    std::string saveTrackingImage = "./tracking/";
    std::string prefix = "tracking_"+std::to_string(tracker_type)+"_"+std::to_string(FLAGS_display)+"_";

    for (int i = 2; i < 10; i++) {
      std::string fileName = rootFolder +"000" +std::to_string(i) +".jpg";
      tracker->track(fileName);
      std::cout << "Frame #" << i  << " out of "
                << 10 << std::endl;

      cv::Mat tracking_image = tracker->getObjectnessCanvas();

      std::string savefilename =
          saveTrackingImage + prefix + std::to_string(1000 + i) + ".png";
      std::cout << "saving to file: " << savefilename << std::endl;
      cv::imwrite(savefilename, tracking_image);
    }
  return 0;
}
