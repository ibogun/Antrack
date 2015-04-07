//
// Created by Ivan Bogun on 4/2/15.
//

#include "ExperimentRunner.h"
#include "../Tracker/Struck.h"
#include <thread>
#include <algorithm>

#include <sstream>
#include <random>
#include <boost/filesystem.hpp>

void ExperimentRunner::runOneThreadOneJob(int startingFrame, cv::Rect initialBox, std::vector<std::string> frameNames,
                        std::string saveName,
                        bool saveResults,
                        bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling, bool scalePrior,
                        std::string kernel, std::string feature,int display) {


    // forward run of the tracker
    Struck forwardTracker = Struck::getTracker(pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
                                               kernel,
                                               feature);



    forwardTracker.display=display;
    Struck backwardTracker = Struck::getTracker(pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
                                                kernel,
                                                feature);

    backwardTracker.display=display;

    std::string initalFrame = frameNames[startingFrame];
    cv::Mat im = cv::imread(initalFrame);

    if (startingFrame > 0) {
        backwardTracker.initialize(im, initialBox);

        for (int i = startingFrame - 1; i >= 0; i--) {
            cv::Mat image = cv::imread(frameNames[i]);
            backwardTracker.track(image);

        }
    }

    if (startingFrame <= frameNames.size() - 1) {
        forwardTracker.initialize(im, initialBox);

        for (int i = startingFrame + 1; i < frameNames.size(); i++) {
            cv::Mat image = cv::imread(frameNames[i]);
            forwardTracker.track(image);

        }

    }

//    std::cout << "Size of the backward Tracker: " << backwardTracker.boundingBoxes.size() << std::endl;
//    std::cout << "Size of the forward Tracker: " << forwardTracker.boundingBoxes.size() << std::endl;
//
//
//    if (forwardTracker.boundingBoxes.size() > 0) {
//        std::cout << "First box forward: " << forwardTracker.boundingBoxes[0] << std::endl;
//    }
//
//    if (backwardTracker.boundingBoxes.size() > 0) {
//        std::cout << "First box backward: " << backwardTracker.boundingBoxes[0] << std::endl;
//    }
    // combine trackers

    std::reverse(std::begin(backwardTracker.boundingBoxes), std::end(backwardTracker.boundingBoxes));

    int j = 0;

    if (startingFrame > 0) {
        j = 1;
    }

    for (; j < forwardTracker.boundingBoxes.size(); ++j) {
        backwardTracker.boundingBoxes.push_back(forwardTracker.boundingBoxes[j]);
    }

    //std::cout << " Total boxes written: " << backwardTracker.boundingBoxes.size() << std::endl;

    if (saveResults) {
        backwardTracker.saveResults(saveName);
    }

    //forwardTracker.reset();
    //backwardTracker.reset();
}


void runOneThreadMultipleJobs(std::vector<std::tuple<int, int, cv::Rect>> &jobs,
                              vector<pair<string, vector<string>>> &vidData, std::string saveName, int from, int to,
                              bool saveResults,
                              bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling,
                              bool scalePrior,
                              std::string kernel, std::string feature) {

    // run jobs from index 'from' to the index 'to', make sure to create proper saveName

    for (int i = from; i < to; ++i) {

        int videoNumber = std::get<0>(jobs[i]);
        int frame = std::get<1>(jobs[i]);
        cv::Rect bb = std::get<2>(jobs[i]);

        std::string videoName = vidData[videoNumber].first;

        std::vector<std::string> frameNames = vidData[videoNumber].second;

        std::stringstream ss;

        ss << saveName << "/" << videoName << "__" << std::to_string(i) << ".dat";


        std::string finalFilename = ss.str();


        if (boost::filesystem::exists(finalFilename)){
            continue;
        }


        ExperimentRunner::runOneThreadOneJob(frame, bb, frameNames, finalFilename, saveResults, pretraining, useFilter, useEdgeDensity,
                           useStraddling, scalePrior, kernel, feature);
    }

}




void ExperimentRunner::run(std::string saveFolder, int n_threads, bool saveResults,
                           bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling,
                           bool scalePrior,
                           std::string kernel, std::string feature) {
    using namespace std;

    // the vector below requires reshuffling
    std::vector<std::tuple<int, int, cv::Rect>> jobs = this->experiment->generateAllBoxesToEvaluate(this->dataset);

    std::cout << "Number of jobs: " << jobs.size() << std::endl;

    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();

    for (int j = 0; j < video_gt_images.size(); ++j) {
        pair<string, vector<string>> p = video_gt_images[j];
        p.first = this->dataset->videos[j];
        video_gt_images[j] = p;
    }


    auto engine = std::default_random_engine{};
    std::shuffle(std::begin(jobs), std::end(jobs), engine);


    std::time_t t1 = std::time(0);


    std::vector<std::thread> th;

    arma::rowvec bounds = arma::linspace<rowvec>(
            0, jobs.size(), MIN(MAX(n_threads, 2), jobs.size()));

    bounds = arma::round(bounds);

    for (int i = 0; i < bounds.size() - 1; i++) {

        /*
         * (std::vector<std::tuple<int, int, cv::Rect>> &jobs,
                              vector<pair<string, vector<string>>> &vidData, std::string saveName, int from, int to,
                              bool saveResults,
                              bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling,
                              bool scalePrior,
                              std::string kernel, std::string feature)
         *
         *
         */


        th.push_back(std::thread(
                runOneThreadMultipleJobs, std::ref(jobs), std::ref(video_gt_images), std::ref(saveFolder),
                std::ref(bounds[i]), std::ref(bounds[i + 1]),
                std::ref(saveResults), std::ref(pretraining),
                std::ref(useFilter), std::ref(useEdgeDensity),
                std::ref(useStraddling), std::ref(scalePrior), std::ref(kernel), std::ref(feature)));
    }

    for (auto &t : th) {
        t.join();
    }


    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;

    std::cout << this->experiment->getInfo() << std::endl;
    std::cout << "Time with threads: " << (t2 - t1) << std::endl;


    std::ofstream out(saveFolder + "/" + "tracker_info.txt");
    out << Struck::getTracker(pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
                              kernel,
                              feature);


    out.close();

    std::ofstream outExperiment(saveFolder + "/" + "experiment_info.txt");
    outExperiment << *this;
    outExperiment.close();

}


void ExperimentRunner::runExample(int video, int startingFrame, std::string saveName, bool saveResults,
                                  bool pretraining,
                                  bool useFilter, bool useEdgeDensity, bool useStraddling, bool scalePrior,
                                  std::string kernel, std::string feature, int display) {


    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();


    pair<string, vector<string>> p = video_gt_images[video];


    vector<string> frames = p.second;

    std::vector<cv::Rect> rects = this->dataset->readGroundTruth(p.first);

    cv::Rect gt = rects[startingFrame];

    runOneThreadOneJob(startingFrame, gt, frames, saveName, saveResults, pretraining, useFilter, useEdgeDensity,
                       useStraddling, scalePrior, kernel, feature,display);

}

std::ostream &operator<<(std::ostream &strm, const ExperimentRunner &f) {

    std::string line = "--------------------------------------------------------\n";
    strm << "Experiment runner\n" << line;


    strm << f.experiment->getInfo();
    strm << line;

    strm << f.dataset->getInfo();

    return strm;

}