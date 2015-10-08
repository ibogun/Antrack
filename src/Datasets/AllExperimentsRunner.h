//
// Created by Ivan Bogun on 4/2/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_ALLEXPERIMENTSRUNNER_H
#define ROBUST_TRACKING_BY_DETECTION_ALLEXPERIMENTSRUNNER_H

#include "ExperimentTemporalRobustness.h"
#include "ExperimentSpatialRobustness.h"
#include "ExperimentDefault.h"
#include "ExperimentRunner.h"

class AllExperimentsRunner {

    Dataset *dataset;


 public:
    AllExperimentsRunner(Dataset* d): dataset(d){}

    void run(std::string safeFolder, int nThreads, bool saveResults,
             bool pretraining, bool useFilter, bool useEdgeDensity,
             bool useStraddling,
             bool scalePrior,
             std::string kernel, std::string feature,
             int updateEveryNFrames,double b,int P, int R, int Q,
             const std::unordered_map<std::string, double>& map,
             int display = 0);

    void runSmall(std::string safeFolder, int nThreads, bool saveResults,
                  bool pretraining, bool useFilter, bool useEdgeDensity,
                  bool useStraddling,
                  bool scalePrior,
                  std::string kernel, std::string feature,
                  int updateEveryNFrames,double b,int P, int R, int Q,
                  std::unordered_map<std::string, double> map,
                  int display=0);


    static void createDirectory(std::string s);
    static void deleteDirectory(std::string s);
};


#endif //ROBUST_TRACKING_BY_DETECTION_ALLEXPERIMENTSRUNNER_H
