//
//  DatasetWu2013.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/7/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "DatasetWu2013.h"
#include <fstream>
#include <regex>


/**
 *  Read ground truth files for Wu2013 dataset
 *
 *  @param fileName folder with ground truth files
 *
 *  @return array of ground truth bounding boxes
 */
std::vector<cv::Rect> DatasetWu2013::readGroundTruth(std::string fileName){
    
    using namespace std;
    std::ifstream infile(fileName);
    string str;
    int num=0;
    std::regex e ("[[:digit:]]+");
    

    vector<cv::Rect> gtRects;
    
    cv::Rect varRect;
    
    int idx=0;
    while (std::getline(infile, str))
    {
        
        
        std::regex_iterator<std::string::iterator> rit ( str.begin(), str.end(), e );
        std::regex_iterator<std::string::iterator> rend;
        
        int idxValues=0;
        while (rit!=rend) {
            
            num=stoi(rit->str());
            
            if (num<0|| num>2000){
                cout<<num<<" READING IS NOT CORRECT"<<endl;
            }
            
            //std::cout << num << std::endl;
            
            switch (idxValues) {
                case 0:
                    varRect.x=num;
                    break;
                case 1:
                    varRect.y=num;
                    break;
                case 2:
                    
                    varRect.width=num;
                    break;
                case 3:
                    varRect.height=num;
                    break;
                default:
                    break;
            }
            
            ++rit;
            ++idxValues;
            
            
        }
        gtRects.push_back(varRect);

        ++idx;
    }
    
    
    return gtRects;
}

/**
 *  Return array of videos each of which is represented by array of images
 *
 *  @param rootFolder root folder of videos
 *
 *  @return array <array<images>> where array<images> is a video
 */
std::vector<std::pair<std::string, std::vector<std::string>>> DatasetWu2013::prepareDataset(std::string rootFolder){
    
    using namespace std;

    vector<pair<string, vector<string>>> video_gt_images;
    
    
    // step #1: list all videos
    vector<string> videos=listSubFolders(rootFolder);
    
    for (int i=0; i<(int)videos.size(); i++) {
        
        // create a string which represents ground truth and image location
        string imgLocation=rootFolder+videos[i]+"/img/";
        string gt=rootFolder+videos[i]+"/groundtruth_rect.txt";
        
        // list all images
        
        vector<string> images=listImages(imgLocation, "jpg");
        
        this->vidToIndex.insert(std::pair<std::string, int>(videos[i],i));
        
        std::pair<string, vector<string>> groundTruth_images;
        groundTruth_images=std::make_pair(gt, images);
        
        video_gt_images.push_back(groundTruth_images);
        
        this->videos.push_back(videos[i]);
        
    }
    
    
    return video_gt_images;
}
