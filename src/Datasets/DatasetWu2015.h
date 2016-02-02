//
// Created by Ivan Bogun on 2/1/16.
//

#ifndef ROBUST_TRACKING_BY_DETECTION_DATASETWU2015_H
#define ROBUST_TRACKING_BY_DETECTION_DATASETWU2015_H

#include "DatasetWu2013.h"
class DatasetWu2015: public DatasetWu2013{

public:
    std::string getInfo() {
        std::stringstream ss;
        ss << "Wu dataset (PAMI 2015) \n";
        ss << "100 videos \n";
        return ss.str();
    }

    ~DatasetWu2015(){}
};


#endif //ROBUST_TRACKING_BY_DETECTION_DATASETWU2015_H
