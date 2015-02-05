//
//  DatasetALOV300.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/26/14.
//
//

#include "DatasetALOV300.h"
#include <regex>
#include <fstream>


/**
 *  Read ALOV300 dataset
 *
 *  @param fileName name of the file containing bounding boxes
 *
 *  @return vector of bounding boxes
 */
std::vector<cv::Rect> DatasetALOV300::readGroundTruth(std::string fileName){
    
    using namespace std;
    
    vector<cv::Rect> gtRects;
    
    cv::Rect varRect;
    
    std::ifstream infile(fileName);
    string str;
    std::regex e ("[[:digit:]]+");
    
    
    int frameIdx=0;
    
    
    while (std::getline(infile, str))
    {
        
        std::istringstream iss(str);
        
        //int frameIdx;
        double x1,y1,x2,y2,x3,y3,x4,y4;
        
        if(!(iss>>frameIdx>>x1>>y1>>x2>>y2>>x3>>y3>>x4>>y4)){break;}

        cv::Rect rect(cvRound(fmin(x2,x1))-1,cvRound(fmin(y1,y3))-1,abs(x1-x2),abs(y3-y1));
        gtRects.push_back(rect);
        
    }
    
    return gtRects;
    
    
}


std::vector<std::pair<std::string,std::vector<std::string>>> DatasetALOV300::prepareDataset(std::string rootFolder){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images;
    
    //
    
    /*
     Assume that rootFolder is
     str="/Users/Ivan/Files/Data/Tracking_alov300/";
     
     where images are in 

     images      = str+"/imagedata++/"+imageType
     groundTruth = str+"/alov300++_rectangleAnnotation_full/"+imageType
     */
    
    // step #1: list all types of videos
    //vector<string> videos           = listSubFolders(rootFolder);

    string datasetFolder            = rootFolder+"/imagedata++/";
    string datasetGroundTruthFolder = rootFolder+"/alov300++_rectangleAnnotation_full/";
    
    vector<string> videoTypes       = listSubFolders(datasetFolder);
    vector<string> groundTruthTypes = listSubFolders(datasetGroundTruthFolder);
    
    for (int i=0; i<(int)videoTypes.size(); i++) {

        string videosSpecificType = datasetFolder+videoTypes[i]+"/";
        string videosType_GT      = datasetGroundTruthFolder+videoTypes[i]+"/";
        
        vector<string> videos     = listSubFolders(videosSpecificType);
        vector<string> videos_gt = listImages(videosType_GT, "ann");
        
        for (int j=0; j<(int)videos.size(); j++) {
            
            string fullVideoPath=videosSpecificType+videos[j]+"/";
            string fullGroundTruthPath=videos_gt[j];
            
            vector<string> images=listImages(fullVideoPath, "jpg");
            
            std::pair<string, vector<string>> groundTruth_images;
            groundTruth_images=std::make_pair(fullGroundTruthPath, images);
            
            video_gt_images.push_back(groundTruth_images);
            
            this->videos.push_back(videos[j]);
            
        }
        
    }
    
    
    return video_gt_images;
    

    
}