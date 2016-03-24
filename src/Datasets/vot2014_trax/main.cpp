//
//  main.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <ctime>
#include <iostream>

#include <fstream>

#include <pthread.h>
#include <stdio.h>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "../../Kernels/AllKernels.h"

#include "../../Features/AllFeatures.h"

#include "../../Datasets/AllDatasets.h"
#include "../../Tracker/AllTrackers.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

//#ifdef TRAX
#include "trax.h"
//#else
//#include "vot.hpp"
//#endif

DEFINE_string(feature, "hogANDhist", "Feature (e.g. raw, hist, haar, hog).");
DEFINE_string(kernel, "int", "Kernel to use (e.g. linear, gauss, int).");
DEFINE_int32(filter, 1, "Use Robust Kalman filter on not.");
DEFINE_int32(updateEveryNframes, 3, "Update tracker every N frames.");
DEFINE_int32(b, 10, "Robust constant.");
DEFINE_int32(P, 10, "P in Robust Kalman filter.");
DEFINE_int32(Q, 13, "Q in Robust Kalman filter.");
DEFINE_int32(R, 13, "R in Robust Kalman filter.");
DEFINE_double(lambda_s, 0.0, "Straddling lambda in ObjDetectorTracker().");
DEFINE_double(lambda_e, 0.5, "Edge density lambda in ObjDetectorTracker().");
DEFINE_double(inner, 0.9, "Inner bounding box for objectness.");
DEFINE_double(straddeling_threshold, 1.5, "Straddeling threshold.");

DEFINE_string(top_feature, "deep", "Top features to use");
DEFINE_string(top_kernel, "linear", "Top kernel to use");

DEFINE_double(lambda_diff, 0.075, "Lambda used in multiple hypothesis.");
DEFINE_int32(MBest, 64, "MBest M.");

DEFINE_string(
    proto_file,
    "/Users/Ivan/Code/Tracking/DeepAntrack/data/imagenet_memory.prototxt",
    "Proto file for the feature extraction using deep ConvNet");
DEFINE_string(conv_deep_weights, "/Users/Ivan/Code/Tracking/DeepAntrack/data/"
                                 "bvlc_reference_caffenet.caffemodel",
              "File with the weights for the deep ConvNet");

#include <boost/program_options.hpp>

namespace po = boost::program_options;

//#ifdef TRAX
int main(int ac, char **av) {
    google::InitGoogleLogging(av[0]);
    gflags::ParseCommandLineFlags(&ac, &av, true);
    try {

        int updateEveryNthFrames = FLAGS_updateEveryNframes;
        double b = FLAGS_b;
        int P = FLAGS_P;
        int Q = FLAGS_Q;
        int R = FLAGS_R;

        bool filter = true;

        std::string feature = FLAGS_feature;
        std::string kernel = FLAGS_kernel;

        bool pretraining = false;
        bool useEdgeDensity = true;
        bool useStraddling = true;
        bool scalePrior = false;

        std::string note_ = "";

        MBestStruck tracker(pretraining, filter, useEdgeDensity, useStraddling,
                            scalePrior, kernel, feature, note_);

        std::unordered_map<std::string, double> map;

        map.insert(std::make_pair("lambda_straddling", FLAGS_lambda_s));
        map.insert(std::make_pair("lambda_edgeness", FLAGS_lambda_e));
        map.insert(std::make_pair("inner", FLAGS_inner));
        map.insert(std::make_pair("straddling_threshold",
                                  FLAGS_straddeling_threshold));

        std::unordered_map<std::string, std::string> featureParamsMap;

        featureParamsMap.insert(std::make_pair("dis_features", "hog"));
        featureParamsMap.insert(std::make_pair("dis_kernel", "linear"));

        featureParamsMap.insert(
            std::make_pair("top_features", FLAGS_top_feature));
        featureParamsMap.insert(std::make_pair("top_kernel", FLAGS_top_kernel));

        featureParamsMap.insert(std::make_pair("proto", FLAGS_proto_file));
        featureParamsMap.insert(
            std::make_pair("weights", FLAGS_conv_deep_weights));

        tracker.setParams(map);
        tracker.setLambda(FLAGS_lambda_diff);
        tracker.setM(FLAGS_MBest);
        tracker.setFeatureParams(featureParamsMap);

        trax_image *img = NULL;
        trax_region *reg = NULL;
        trax_region *mem = NULL;

        trax_handle *trax;
        trax_configuration config;
        config.format_region = TRAX_REGION_POLYGON;
        // TRAX_REGION_RECTANGLE;
        config.format_image = TRAX_IMAGE_PATH;

        trax = trax_server_setup_standard(config, NULL);

        bool run = true;

        while (run) {

            int tr = trax_server_wait(trax, &img, &reg, NULL);

            if (tr == TRAX_INITIALIZE) {

                float x, y, width, height;
                // trax_region_get_rectangle(reg, &x, &y, &width, &height);

                std::vector<float> record;
                for (int j = 0; j < 4; ++j) {
                    float x, y;

                    trax_region_get_polygon_point(reg, j, &x, &y);

                    record.push_back(x);
                    record.push_back(y);
                }

                cv::Rect rect = DatasetVOT2015::constructRectangle(record);
                cv::Mat image = cv::imread(trax_image_get_path(img));

                if (rect.x + rect.width >= image.cols) {
                    rect.width = image.cols - rect.x - 1;
                }

                if (rect.y + rect.height >= image.rows) {
                    rect.height = image.rows - rect.y - 1;
                }

                tracker.initialize(image, rect);
                trax_server_reply(trax, reg, NULL);

            } else if (tr == TRAX_FRAME) {

                cv::Mat image = cv::imread(trax_image_get_path(img));

                cv::Rect rect = tracker.track(image);

                trax_region *result = trax_region_create_rectangle(
                    static_cast<double>(rect.x), static_cast<double>(rect.y),
                    static_cast<double>(rect.width),
                    static_cast<double>(rect.height));

                trax_server_reply(trax, result, NULL);

                trax_region_release(&result);

            } else {

                run = false;
            }

            if (img)
                trax_image_release(&img);
            if (reg)
                trax_region_release(&reg);
        }

        if (mem)
            trax_region_release(&mem);

        trax_cleanup(&trax);

        return 0;

    } catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        cerr << "Exception of unknown type!\n";
    }
}
