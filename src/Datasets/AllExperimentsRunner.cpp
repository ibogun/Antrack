//
// Created by Ivan Bogun on 4/2/15.
//

#include "AllExperimentsRunner.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <vector>
#include <thread>
#include "../Tracker/Struck.h"
#include "../Tracker/ObjDetectorStruck.h"
#include <sstream>
#include <random>
#include "armadillo"
//#include <boost/system/>


void AllExperimentsRunner::createDirectory(std::string s) {
    const char *path = s.c_str();
    boost::filesystem::path dir(path);

    if (!(boost::filesystem::exists(dir))){

        boost::filesystem::create_directory(dir);
    }

}

void AllExperimentsRunner::deleteDirectory(std::string s) {
    const char *path = s.c_str();
    boost::filesystem::path dir(path);
    boost::filesystem::remove_all(dir);
}

void saveData(Experiment* e, std::string saveFolder, bool pretraining,
              bool useFilter, bool useEdgeDensity,
              bool useStraddling,
              bool scalePrior, std::string kernel, std::string feature) {

    std::ofstream out(saveFolder + "/" + "tracker_info.txt");
    out << ObjDetectorStruck::getTracker(pretraining, useFilter, useEdgeDensity,
                                         useStraddling, scalePrior,
                              kernel,
                              feature,"");
    out.close();

    std::ofstream outExperiment(saveFolder + "/" + "experiment_info.txt");
    outExperiment << e->getInfo();
    outExperiment.close();
}


void runOneThreadMultipleJobs(std::vector<std::tuple<std::string, int, int,
                              cv::Rect>> &jobs,
                              std::vector<std::pair<std::string,
                              std::vector<std::string>>> &vidData,
                              int from,
                              int to,
                              bool saveResults,
                              bool pretraining,
                              bool useFilter,
                              bool useEdgeDensity,
                              bool useStraddling,
                              bool scalePrior,
                              std::string
                              kernel,
                              std::string feature,int updateEveryNFrames,
                              double b, int P, int R, int Q,
                              const std::unordered_map<std::string, double>& map,
                              int display) {

// run jobs from index 'from' to the index 'to', make sure to create proper saveName

    for (int i = from; i < to; i++) {


        int videoNumber = std::get<1>(jobs[i]);
        int frame = std::get<2>(jobs[i]);
        cv::Rect bb = std::get<3>(jobs[i]);
        std::string saveName = std::get<0>(jobs[i]);

        std::string videoName = vidData[videoNumber].first;

        std::vector<std::string> frameNames = vidData[videoNumber].second;

        std::stringstream ss;

        ss << saveName << "/" << videoName <<"_sframe="<<std::to_string(frame)
           <<"__" << std::to_string(i) << ".dat";

        std::string finalFilename = ss.str();

        std::cout<< "Current video: " <<  videoName << std::endl;;
        ExperimentRunner::
            runOneThreadOneJob(frame, bb, frameNames,
                               finalFilename, saveResults,
                               pretraining, useFilter,
                               useEdgeDensity,
                               useStraddling, scalePrior, kernel,
                               feature, updateEveryNFrames, b, P,
                               R, Q,
                               map,
                               display);
    }
}

void AllExperimentsRunner::runSmall(std::string saveFolder, int nThreads,
                                    bool saveResults, bool pretraining,
                                    bool useFilter,
                                    bool useEdgeDensity, bool useStraddling,
                                    bool scalePrior, std::string kernel,
                                    std::string feature,int updateEveryNFrames,
                                    double b, int P, int R, int Q,
                                    std::unordered_map<std::string, double> map, int display) {

    ExperimentDefault ed;

    std::string saveFolderDefault = saveFolder + "/default/";

    createDirectory(saveFolderDefault);


    std::vector<std::tuple<int, int, cv::Rect>> ed_boxes = ed.generateAllBoxesToEvaluate(this->dataset);



    std::vector<std::tuple<std::string, int, int, cv::Rect>> jobs;

    for (int i = 0; i < ed_boxes.size(); ++i) {

        int video = std::get<0>(ed_boxes[i]);
        int frame = std::get<1>(ed_boxes[i]);
        cv::Rect box = std::get<2>(ed_boxes[i]);
        auto t = std::make_tuple(saveFolderDefault, video, frame, box);

        jobs.push_back(t);
    }




//    runner1.run(saveFolderDefault, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling,
//                scalePrior, kernel, feature);
//    runner2.run(saveFolderSRE, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
//                kernel, feature);
//    runner3.run(saveFolderTRE, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
//                kernel, feature);

    using namespace std;
    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();

    for (int j = 0; j < video_gt_images.size(); ++j) {
        pair<string, vector<string>> p = video_gt_images[j];
        p.first = this->dataset->videos[j];


        vector<string> p_second=p.second;

        video_gt_images[j] = p;
    }


    // to make sure same set of boxes is generates all the time.
    std::srand(143);
    auto engine = std::default_random_engine{};
    //engine.seed(100);
    std::shuffle(std::begin(jobs), std::end(jobs), engine);




    std::time_t t1 = std::time(0);


    std::vector<std::thread> th;

    arma::rowvec bounds = arma::linspace<arma::rowvec>(
            0, jobs.size(), MIN(MAX(nThreads, 2), jobs.size()));

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

        if (nThreads == 1){

        runOneThreadMultipleJobs(jobs,video_gt_images,bounds[i],bounds[i + 1],
        saveResults,pretraining, useFilter, useEdgeDensity, useStraddling , scalePrior, kernel,
                                 feature, updateEveryNFrames, b, P, R, Q, map, display);
        } else {
        th.push_back(std::thread(
                runOneThreadMultipleJobs, std::ref(jobs),
                std::ref(video_gt_images),
                std::ref(bounds[i]), std::ref(bounds[i + 1]),
                std::ref(saveResults), std::ref(pretraining),
                std::ref(useFilter), std::ref(useEdgeDensity),
                std::ref(useStraddling), std::ref(scalePrior), std::ref(kernel),
                std::ref(feature),
                std::ref(updateEveryNFrames), std::ref(b), std::ref(P),
                std::ref(R), std::ref(Q), std::ref(map), std::ref(display)));
        }
    }

    if (nThreads != 1) {
        for (auto &t : th) {
            t.join();
        }
    }


    saveData(&ed, saveFolderDefault, pretraining, useFilter, useEdgeDensity,
             useStraddling,
             scalePrior, kernel, feature);


}


void AllExperimentsRunner::run(std::string saveFolder, int nThreads,
                               bool saveResults, bool pretraining,
                               bool useFilter,
                               bool useEdgeDensity, bool useStraddling,
                               bool scalePrior, std::string kernel,
                               std::string feature,int updateEveryNFrames,
                               double b, int P, int R, int Q,
                               const std::unordered_map<std::string, double>& map,int display) {

    //ExperimentDefault ed;
    ExperimentSpatialRobustness es;
    ExperimentTemporalRobustness et;


//    ExperimentRunner runner1(&ed, this->dataset);
//    ExperimentRunner runner2(&es, this->dataset);
//    ExperimentRunner runner3(&et, this->dataset);

    //std::string saveFolderDefault = saveFolder + "/default/";
    std::string saveFolderSRE = saveFolder + "/SRE/";
    std::string saveFolderTRE = saveFolder + "/TRE/";


    //deleteDirectory(saveFolderDefault);
    //deleteDirectory(saveFolderSRE);
    //deleteDirectory(saveFolderTRE);

    //createDirectory(saveFolderDefault);
    createDirectory(saveFolderSRE);
    createDirectory(saveFolderTRE);



    //std::vector<std::tuple<int, int, cv::Rect>> ed_boxes = ed.generateAllBoxesToEvaluate(this->dataset);
    std::vector<std::tuple<int, int, cv::Rect>> es_boxes =
        es.generateAllBoxesToEvaluate(this->dataset);
    std::vector<std::tuple<int, int, cv::Rect>> et_boxes =
        et.generateAllBoxesToEvaluate(this->dataset);


    std::vector<std::tuple<std::string, int, int, cv::Rect>> jobs;

//    for (int i = 0; i < ed_boxes.size(); ++i) {
//
//        int video = std::get<0>(ed_boxes[i]);
//        int frame = std::get<1>(ed_boxes[i]);
//        cv::Rect box = std::get<2>(ed_boxes[i]);
//        auto t = std::make_tuple(saveFolderDefault, video, frame, box);
//
//        jobs.push_back(t);
//    }

    for (int i = 0; i < es_boxes.size(); ++i) {

        int video = std::get<0>(es_boxes[i]);
        int frame = std::get<1>(es_boxes[i]);
        cv::Rect box = std::get<2>(es_boxes[i]);
        auto t = std::make_tuple(saveFolderSRE, video, frame, box);

        jobs.push_back(t);
    }


    for (int i = 0; i < et_boxes.size(); ++i) {

        int video = std::get<0>(et_boxes[i]);
        int frame = std::get<1>(et_boxes[i]);
        cv::Rect box = std::get<2>(et_boxes[i]);
        auto t = std::make_tuple(saveFolderTRE, video, frame, box);

        jobs.push_back(t);
    }



//    runner1.run(saveFolderDefault, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling,
//                scalePrior, kernel, feature);
//    runner2.run(saveFolderSRE, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
//                kernel, feature);
//    runner3.run(saveFolderTRE, nThreads, saveResults, pretraining, useFilter, useEdgeDensity, useStraddling, scalePrior,
//                kernel, feature);

    using namespace std;
    vector<pair<string, vector<string>>> video_gt_images =
            this->dataset->prepareDataset();

    for (int j = 0; j < video_gt_images.size(); ++j) {
        pair<string, vector<string>> p = video_gt_images[j];
        p.first = this->dataset->videos[j];


        vector<string> p_second=p.second;

        video_gt_images[j] = p;



    }


    // to make sure same set of boxes is generates all the time.
    std::srand(0);
    auto engine = std::default_random_engine{};
    std::shuffle(std::begin(jobs), std::end(jobs), engine);




    std::time_t t1 = std::time(0);


    std::vector<std::thread> th;

    arma::rowvec bounds = arma::linspace<arma::rowvec>(
            0, jobs.size(), MIN(MAX(nThreads, 2), jobs.size()));

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
                runOneThreadMultipleJobs, std::ref(jobs), std::ref(video_gt_images),
                std::ref(bounds[i]), std::ref(bounds[i + 1]),
                std::ref(saveResults), std::ref(pretraining),
                std::ref(useFilter), std::ref(useEdgeDensity),
                std::ref(useStraddling), std::ref(scalePrior), std::ref(kernel), std::ref(feature),
                std::ref(updateEveryNFrames),std::ref(b),std::ref(P),
                std::ref(R),std::ref(Q), std::ref(map),
                std::ref(display)));
    }

    for (auto &t : th) {
        t.join();
    }


//    saveData(&ed, saveFolderDefault, pretraining, useFilter, useEdgeDensity,
//             useStraddling,
//             scalePrior, kernel, feature);

    saveData(&es, saveFolderSRE, pretraining, useFilter, useEdgeDensity,
             useStraddling,
             scalePrior, kernel, feature);

    saveData(&et, saveFolderTRE, pretraining, useFilter, useEdgeDensity,
             useStraddling,
             scalePrior, kernel, feature);

}




