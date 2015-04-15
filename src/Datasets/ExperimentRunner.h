//
// Created by Ivan Bogun on 4/2/15.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_EXPERIMENTRUNNER_H
#define ROBUST_TRACKING_BY_DETECTION_EXPERIMENTRUNNER_H

#include "Experiment.h"
#include "Dataset.h"

class ExperimentRunner {

    Experiment *experiment;
    Dataset *dataset;

    friend std::ostream& operator<<(std::ostream&,const  ExperimentRunner&);
public:

    ExperimentRunner(Experiment *e, Dataset *d) : experiment(e), dataset(d) { }

    void run(std::string safeFolder, int nThreads, bool saveResults,
             bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling,
             bool scalePrior,
             std::string kernel, std::string feature, double b,int P, int R, int Q);

    void runExample(int video,int startingFrame,
                    std::string saveName,
                    bool saveResults,
                    bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling, bool scalePrior,
                    std::string kernel, std::string feature,double b, int display);


    static void runOneThreadOneJob(int startingFrame, cv::Rect initialBox, std::vector<std::string> frameNames,
                            std::string saveName,
                            bool saveResults,
                            bool pretraining, bool useFilter, bool useEdgeDensity, bool useStraddling, bool scalePrior,
                            std::string kernel, std::string feature,double b, int P, int R, int Q,int display=0);
};


#endif //ROBUST_TRACKING_BY_DETECTION_EXPERIMENTRUNNER_H
