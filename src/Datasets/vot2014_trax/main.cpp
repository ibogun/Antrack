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

#include "../../Kernels/RBFKernel.h"
#include "../../Kernels/IntersectionKernel.h"
#include "../../Kernels/IntersectionKernel_fast.h"
#include "../../Kernels/ApproximateKernel.h"
#include "../../Kernels/LinearKernel.h"

#include "../../Features/RawFeatures.h"
#include "../../Features/Haar.h"
#include "../../Features/Histogram.h"
#include "../../Features/HoG.h"

#include "../../Tracker/LocationSampler.h"
#include "../../Tracker/OLaRank_old.h"
#include "../../Tracker/Struck.h"

#include "../../Datasets/DataSetWu2013.h"
#include "../../Datasets/DatasetALOV300.h"
#include "../../Datasets/DatasetVOT2014.h"

#include <pthread.h>
#include <thread>

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

//#ifdef TRAX
#include "trax.h"
//#else
//#include "vot.hpp"
//#endif




#include <boost/program_options.hpp>

namespace po = boost::program_options;


//#ifdef TRAX
int main( int ac, char** av)
{



    namespace po = boost::program_options;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")(
                "feature", po::value<std::string>(), "Feature (e.g. raw, hist, haar, hog)")(
                "kernel", po::value<std::string>(), "Kernel (e.g. linear, gauss, int)")(
                "filter",po::value<bool>()," Filter 1-on, 0-off")(
                "b", po::value<double>(), " Robust constant")(
                "P", po::value<int>(), "P")(
                "Q", po::value<int>(), "Q")(
                "R", po::value<int>(), "R");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }


        double b = 10;
        int P = 3;
        int Q = 5;
        int R = 5;
        bool useFilter;

        std::string feature = "raw";
        std::string kernel = "linear";

        if (vm.count("filter")) {
            cout << "Filter is: " << vm["filter"].as<bool>() << ".\n";
            useFilter = vm["filter"].as<bool>();
        }

        if (vm.count("feature")) {
            cout << "Feature is: " << vm["feature"].as<std::string>() << ".\n";
            feature = vm["feature"].as<std::string>();
        }

        if (vm.count("kernel")) {
            cout << "Kernel is: " << vm["kernel"].as<std::string>() << ".\n";
            kernel = vm["kernel"].as<std::string>();
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

        if (vm.count("b")) {
            cout << "Robust constant is is: " << vm["b"].as<double>() << ".\n";
            b = vm["b"].as<double>();
        }


        bool pretraining=false;
        bool useEdgeDensity=false;
        bool useStraddling=false;
        bool scalePrior=false;

        std::string note_="";

        Struck tracker=Struck::getTracker(pretraining,useFilter,useEdgeDensity,useStraddling,scalePrior,kernel,feature,note_);

        trax_image* img = NULL;
        trax_region* reg = NULL;
        trax_region* mem = NULL;

        trax_handle* trax;
        trax_configuration config;
        config.format_region = TRAX_REGION_POLYGON;
                //TRAX_REGION_RECTANGLE;
        config.format_image = TRAX_IMAGE_PATH;

        trax = trax_server_setup_standard(config, NULL);

        bool run = true;

        while(run)
        {

            int tr = trax_server_wait(trax, &img, &reg, NULL);

            if (tr == TRAX_INITIALIZE) {

                cv::Rect rect;
                float x, y, width, height;
                //trax_region_get_rectangle(reg, &x, &y, &width, &height);


                float center_x=0;
                float center_y=0;

                float left_x,right_x,bottom_y,top_y;
                for (int j = 0; j < 4; ++j) {
                    float x,y;

                    trax_region_get_polygon_point(reg,j,&x,&y);

                    center_x+=x;
                    center_y+=y;

                    if (j==0){
                        left_x=x;
                        top_y=y;
                    }

                    if (j==3){
                        right_x=x;
                    }

                    if (j==2){
                        bottom_y=y;
                    }
                }

                center_x=center_x/4;
                center_y=center_y/4;

                width=abs(right_x-left_x);
                height=abs(bottom_y-top_y);

                rect.x=round(center_x-width/2);
                rect.y=round(center_y-height/2);
                rect.width=round(width);
                rect.height=round(height);
                //rect.x = round(x); rect.y = round(y); rect.width = round(x + width) - rect.x; rect.height = round(y + height) - rect.y;

//            Struck tracker=Struck::getTracker(pretraining,useFilter,useEdgeDensity,useStraddling,scalePrior,kernelSTR,featureSTR,note_);
                cv::Mat image = cv::imread(trax_image_get_path(img));
                tracker.initialize(image, rect,b,P,R,Q);

                trax_server_reply(trax, reg, NULL);

            } else if (tr == TRAX_FRAME) {

                cv::Mat image = cv::imread(trax_image_get_path(img));

                cv::Rect rect = tracker.track(image);

                trax_region* result = trax_region_create_rectangle(rect.x, rect.y, rect.width, rect.height);

                trax_server_reply(trax, result, NULL);

                trax_region_release(&result);

            } else {

                run = false;

            }

            if (img) trax_image_release(&img);
            if (reg) trax_region_release(&reg);

        }

        if (mem) trax_region_release(&mem);

        trax_cleanup(&trax);

        return 0;

    }
    catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        cerr << "Exception of unknown type!\n";
    }




}
