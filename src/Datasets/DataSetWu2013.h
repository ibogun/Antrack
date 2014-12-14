//
//  DataSetWu2013.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/7/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__DataSetWu2013__
#define __Robust_Struck__DataSetWu2013__

#include <stdio.h>
#include "Dataset.h"

class DataSetWu2013:public Dataset{
    

public:

    std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder);
    std::vector<cv::Rect> readGroundTruth(std::string);
    
};

#endif /* defined(__Robust_Struck__DataSetWu2013__) */
