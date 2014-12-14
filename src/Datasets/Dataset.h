//
//  Dataset.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/7/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Dataset__
#define __Robust_Struck__Dataset__

#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Dataset {
    

public:
    
    std::vector<std::string> listSubFolders(std::string folder);
    
    std::vector<std::string> listImages(std::string folder,std::string format);
    
    virtual std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder)=0;
    
    virtual std::vector<cv::Rect> readGroundTruth(std::string)=0;
};

#endif /* defined(__Robust_Struck__Dataset__) */