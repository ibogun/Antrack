//
// Created by Ivan Bogun on 4/2/15.
//

#include <thread>
#include <algorithm>

#include <sstream>
#include <random>
#include <boost/filesystem.hpp>
#include "ExperimentRunner.h"



#include "../Tracker/Struck.h"
#include "../Tracker/ObjDetectorStruck.h"
#include "../Tracker/FilterBadBoxesStruck.h"



void ExperimentRunner::
runOneThreadOneJob(int startingFrame,
                   cv::Rect initialBox,
                   std::vector<std::string> frameNames,
                   std::string saveName,
                   bool saveResults,
                   bool pretraining, bool useFilter,
                   bool useEdgeDensity,
                   bool useStraddling, bool scalePrior,
                   std::string kernel,
                   std::string feature,
                   int updateEveryNFrames,
                   double b, int P, int R, int Q,
                   const std::unordered_map<std::string, double>& map,
                   int display) {
    // forward run of the tracker

    std::string note = "Objectness tracker";

    Struck* forwardTracker = nullptr;
    Struck* backwardTracker = nullptr;

    int tracker_type = map.find("tracker_type")->second;
    if (tracker_type == 0) {
        forwardTracker = new
                Struck(pretraining, useFilter,
                       useEdgeDensity, useStraddling,
                       scalePrior,
                       kernel,
                       feature, note);

        backwardTracker =new
                Struck(pretraining, useFilter,
                       useEdgeDensity, useStraddling,
                       scalePrior,
                       kernel,
                       feature, note);
    }

    if (tracker_type == 1) {
        forwardTracker = new
                ObjDetectorStruck(pretraining, useFilter,
                                  useEdgeDensity, useStraddling,
                                  scalePrior,
                                  kernel,
                                  feature, note);
        backwardTracker = new
                ObjDetectorStruck(pretraining, useFilter,
                                  useEdgeDensity, useStraddling,
                                  scalePrior,
                                  kernel,
                                  feature, note);

        std::cout << "ObjDetectorStruck is being used!" << std::endl;

    }


    if (tracker_type == 2) {
        forwardTracker = new
                FilterBadBoxesStruck(pretraining, useFilter,
                                     useEdgeDensity, useStraddling,
                                     scalePrior,
                                     kernel,
                                     feature, note);
        backwardTracker = new
                FilterBadBoxesStruck(pretraining, useFilter,
                                     useEdgeDensity, useStraddling,
                                     scalePrior,
                                     kernel,
                                     feature, note);
    }

    forwardTracker->setParams(map);
    backwardTracker->setParams(map);

    forwardTracker->display = display;
    backwardTracker->display = display;

    std::string initalFrame = frameNames[startingFrame];
    cv::Mat im = cv::imread(initalFrame);

    // TODO: FIX THIS


    if (startingFrame > 0) {
        backwardTracker->initialize(im, initialBox, updateEveryNFrames, b, P,
                                   R, Q);
        // backwardTracker.initialize(im, initialBox);
        for (int i = startingFrame - 1; i >= 0; i--) {
            // cv::Mat image = cv::imread(frameNames[i]);
            // backwardTracker.track(image);
            backwardTracker->track(frameNames[i]);
        }
    }

    clock_t t1;
    if (display != 0) {
        t1 = clock();
    }
    if (startingFrame <= frameNames.size() - 1) {
        forwardTracker->initialize(im, initialBox, updateEveryNFrames, b, P,
                                  R, Q);
        // forwardTracker.initialize(im, initialBox);
        for (int i = startingFrame + 1; i < frameNames.size(); i++) {
            cv::Mat image = cv::imread(frameNames[i]);
            forwardTracker->track(frameNames[i]);

            std::cout<< frameNames[i] << std::endl;
            // forwardTracker.track(frameNames[i]);
            if (display != 0) {
                clock_t t2 = clock();
                double timeSec = (t2 - t1) / static_cast<double>(
                    CLOCKS_PER_SEC);
                timeSec = (i + 1) / timeSec;
                std::cout << "FPS: " << timeSec << " frame " << i << " / " <<
                    frameNames.size() << "  " <<
                    forwardTracker->getBoundingBoxes()[i - startingFrame - 1]
                          << std::endl;
            }
        }
    }

    // combine trackers

    std::reverse(std::begin(backwardTracker->boundingBoxes),
                 std::end(backwardTracker->boundingBoxes));

    int j = 0;

    if (startingFrame > 0) {
        j = 1;
    }

    for (; j < forwardTracker->boundingBoxes.size(); ++j) {
        backwardTracker->boundingBoxes.push_back(
            forwardTracker->boundingBoxes[j]);
    }

    if (saveResults) {
        backwardTracker->saveResults(saveName);
    }

    delete forwardTracker;
    delete backwardTracker;
}


void runOneThreadMultipleJobs(std::vector<std::tuple<int, int, cv::Rect>> &jobs,
                              vector<pair<string, vector<string>>> &vidData,
                              std::string saveName, int from, int to,
                              bool saveResults,
                              bool pretraining, bool useFilter,
                              bool useEdgeDensity, bool useStraddling,
                              bool scalePrior,
                              std::string kernel, std::string feature,
                              int updateEveryNFrames,double b, int P,
                              int R, int Q,
                              const std::unordered_map<std::string, double>& map) {

    // run jobs from index 'from' to the index 'to', make sure to
    //create proper saveName

    for (int i = from; i < to; ++i) {

        int videoNumber = std::get<0>(jobs[i]);
        int frame = std::get<1>(jobs[i]);
        cv::Rect bb = std::get<2>(jobs[i]);

        std::string videoName = vidData[videoNumber].first;

        std::vector<std::string> frameNames = vidData[videoNumber].second;

        std::stringstream ss;

        ss << saveName << "/" << videoName << "__"
           << std::to_string(i) << ".dat";


        std::string finalFilename = ss.str();


        if (boost::filesystem::exists(finalFilename)){
            continue;
        }


        ExperimentRunner::runOneThreadOneJob(frame, bb, frameNames,
                                             finalFilename, saveResults,
                                             pretraining, useFilter,
                                             useEdgeDensity,
                                             useStraddling, scalePrior, kernel,
                                             feature,updateEveryNFrames,
                                             b, P, R,Q,
                                             map);
    }

}




void ExperimentRunner::run(std::string saveFolder, int n_threads,
                           bool saveResults,
                           bool pretraining, bool useFilter,
                           bool useEdgeDensity, bool useStraddling,
                           bool scalePrior,
                           std::string kernel, std::string feature,
                           int updateEveryNFrames, double b,int P,
                           int R, int Q,
                           const std::unordered_map<std::string, double>& map) {
    using namespace std;

    // the vector below requires reshuffling
    std::vector<std::tuple<int, int, cv::Rect>> jobs =
        this->experiment->generateAllBoxesToEvaluate(this->dataset);

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
        th.push_back(std::thread(
                runOneThreadMultipleJobs, std::ref(jobs),
                std::ref(video_gt_images), std::ref(saveFolder),
                std::ref(bounds[i]), std::ref(bounds[i + 1]),
                std::ref(saveResults), std::ref(pretraining),
                std::ref(useFilter), std::ref(useEdgeDensity),
                std::ref(useStraddling), std::ref(scalePrior),
                std::ref(kernel), std::ref(feature),
                std::ref(updateEveryNFrames),
                std::ref(b), std::ref(P),
                std::ref(R), std::ref(Q),
                std::ref(map)));
    }

    for (auto &t : th) {
        t.join();
    }


    std::time_t t2 = std::time(0);
    // std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;

    std::cout << this->experiment->getInfo() << std::endl;
    std::cout << "Time with threads: " << (t2 - t1) << std::endl;


    std::ofstream out(saveFolder + "/" + "tracker_info.txt");
    out << Struck::getTracker(pretraining, useFilter, useEdgeDensity,
                              useStraddling, scalePrior,
                              kernel,
                              feature);


    out.close();

    std::ofstream outExperiment(saveFolder + "/" + "experiment_info.txt");
    outExperiment << *this;
    outExperiment.close();

}


void ExperimentRunner::runExample(int video, int startingFrame,int endingFrame,
                                  std::string saveName, bool saveResults,
                                  bool pretraining,
                                  bool useFilter, bool useEdgeDensity,
                                  bool useStraddling, bool scalePrior,
                                  std::string kernel,
                                  std::string feature,double b,
                                  const std::unordered_map<std::string, double>& map,
                                  int display) {


    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();


    pair<string, vector<string>> p = video_gt_images[video];


    vector<string> frames = p.second;

    frames.resize(endingFrame);

    std::vector<cv::Rect> rects = this->dataset->readGroundTruth(p.first);

    cv::Rect gt = rects[startingFrame];

    int P=10;
    int R=13;
    int Q=13;
    int updateEveryNFrames=3;

    runOneThreadOneJob(startingFrame, gt, frames, saveName, saveResults,
                       pretraining, useFilter, useEdgeDensity,
                       useStraddling, scalePrior, kernel, feature,
                       updateEveryNFrames,b,P,R,Q, map,
                       display);

}

std::ostream &operator<<(std::ostream &strm, const ExperimentRunner &f) {

    std::string line = "--------------------------------------------------------\n";
    strm << "Experiment runner\n" << line;


    strm << f.experiment->getInfo();
    strm << line;

    strm << f.dataset->getInfo();

    return strm;

}
