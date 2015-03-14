//
//  Experiment.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#ifndef __Robust_tracking_by_detection__Experiment__
#define __Robust_tracking_by_detection__Experiment__

#include <stdio.h>
#include "Dataset.h"
#include "../Tracker/Struck.h"


class Experiment {
    Dataset* dataset;
    
public:
    
    Experiment(Dataset* d){
        this->dataset=d;
    };
    
    
};

#endif /* defined(__Robust_tracking_by_detection__Experiment__) */
