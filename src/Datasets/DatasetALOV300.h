//
//  DatasetALOV300.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/26/14.
//
//

#ifndef __Robust_tracking_by_detection__DatasetALOV300__
#define __Robust_tracking_by_detection__DatasetALOV300__

#include <stdio.h>
#include "Dataset.h"

class DatasetALOV300:public Dataset{
    
    
public:
    //static std::vector<std::string> listSubFolders(std::string folder);
    std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder);
    std::vector<cv::Rect> readGroundTruth(std::string);

    std::string getInfo(){
        std::stringstream ss;

        ss<<"ALOV 300 dataset (PAMI 2014)\n";
        ss<<"108 videos\n";
        return ss.str();
    }
    
};
#endif /* defined(__Robust_tracking_by_detection__DatasetALOV300__) */
