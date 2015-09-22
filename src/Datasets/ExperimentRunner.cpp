//
// Created by Ivan Bogun on 4/2/15.
//

#include "ExperimentRunner.h"
#include "../Tracker/Struck.h"
#include "../Tracker/ObjDetectorStruck.h"
#include <thread>
#include <algorithm>

#include <sstream>
#include <random>
#include <boost/filesystem.hpp>

void ExperimentRunner::runOneThreadOneJob(int startingFrame,
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
                                          double b,int P, int R, int Q,
                                          double lambda,
                                          double straddeling_threshold,
                                          int display) {


    // forward run of the tracker

    std::string note = "Objectness tracker";

    ObjDetectorStruck  forwardTracker =
        ObjDetectorStruck::getTracker(pretraining, useFilter,
                                      useEdgeDensity, useStraddling,
                                      scalePrior,
                                      kernel,
                                      feature, note);

    forwardTracker.setLambda(lambda);
    forwardTracker.setMinStraddeling(straddeling_threshold);

    forwardTracker.display=display;


    ObjDetectorStruck  backwardTracker =
        ObjDetectorStruck::getTracker(pretraining, useFilter,
                                      useEdgeDensity, useStraddling,
                                      scalePrior,
                                      kernel,
                                      feature, note);



    backwardTracker.setLambda(lambda);
    backwardTracker.setMinStraddeling(straddeling_threshold);

    backwardTracker.display=display;

    std::string initalFrame = frameNames[startingFrame];
    cv::Mat im = cv::imread(initalFrame);



    if (startingFrame > 0) {
        backwardTracker.initialize(im, initialBox,updateEveryNFrames,b,P,R,Q);




        for (int i = startingFrame - 1; i >= 0; i--) {
            cv::Mat image = cv::imread(frameNames[i]);
            backwardTracker.track(image);


        }
    }

    clock_t t1;
    if (display!=0){
        t1=clock();
    }
    if (startingFrame <= frameNames.size() - 1) {
        forwardTracker.initialize(im, initialBox,updateEveryNFrames,b,P,R,Q);

        for (int i = startingFrame + 1; i < frameNames.size(); i++) {
            cv::Mat image = cv::imread(frameNames[i]);
            forwardTracker.track(image);

            if (display!=0){
                clock_t t2=clock();

                double timeSec = (t2 - t1) / static_cast<double>(
                    CLOCKS_PER_SEC );
                timeSec=(i+1)/timeSec;

                std::cout<<"FPS: "<<timeSec<<" frame "<<i<<" / "<<
                    frameNames.size() <<std::endl;
            }
        }

    }

    // combine trackers

    std::reverse(std::begin(backwardTracker.boundingBoxes),
                 std::end(backwardTracker.boundingBoxes));

    int j = 0;

    if (startingFrame > 0) {
        j = 1;
    }

    for (; j < forwardTracker.boundingBoxes.size(); ++j) {
        backwardTracker.boundingBoxes.push_back(
            forwardTracker.boundingBoxes[j]);
    }

    if (saveResults) {
        backwardTracker.saveResults(saveName);
    }

    //forwardTracker.reset();
    //backwardTracker.reset();
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
                              double lambda,
                              double straddeling_threshold) {

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
                                             lambda,
                                             straddeling_threshold);
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
                           double lambda,
                           double straddeling_threshold) {
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
                std::ref(b),std::ref(P),
                std::ref(R),std::ref(Q),
                std::ref(lambda), std::ref(straddeling_threshold)));
    }

    for (auto &t : th) {
        t.join();
    }


    std::time_t t2 = std::time(0);
    //std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;

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
                                  double lambda,
                                  double straddeling_threshold,
                                  int display) {


    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();


    pair<string, vector<string>> p = video_gt_images[video];


    vector<string> frames = p.second;

    frames.resize(endingFrame);

    std::vector<cv::Rect> rects = this->dataset->readGroundTruth(p.first);

    cv::Rect gt = rects[startingFrame];

    int P=3;
    int R=5;
    int Q=5;
    int updateEveryNFrames=5;

    runOneThreadOneJob(startingFrame, gt, frames, saveName, saveResults,
                       pretraining, useFilter, useEdgeDensity,
                       useStraddling, scalePrior, kernel, feature,
                       updateEveryNFrames,b,P,R,Q, lambda,
                       straddeling_threshold,
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
