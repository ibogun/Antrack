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
#include <unordered_map>

typedef std::unordered_map<std::string, int> hashtable;
class Dataset {


    std::string root_folder;
public:
    hashtable vidToIndex;
    std::vector<std::string> videos;


    void setRootFolder(std::string root){
        this->root_folder=root;
    }
    static std::vector<std::string> listSubFolders(std::string folder);
    //static std::vector<std::string> listOnlySubfolders(std::string folder);
    
    static std::vector<std::string> listImages(std::string folder,std::string format);
    
    virtual std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(std::string rootFolder)=0;

    std::vector<std::pair<std::string, std::vector<std::string>>> prepareDataset(){
        return this->prepareDataset(this->root_folder);
    }

    virtual std::vector<cv::Rect> readGroundTruth(std::string)=0;
    virtual std::string getInfo()=0;
    virtual ~Dataset(){}
};

#endif /* defined(__Robust_Struck__Dataset__) */
