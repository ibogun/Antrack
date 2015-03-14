//
//  Experiment.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#include "Experiment.h"



//void runTrackerOnDatasetPart(cv::vector<std::pair<std::string, cv::vector<std::string>>>& video_gt_images,Dataset* dataset,
//                             int from, int to,std::string saveFolder, bool saveResults, bool fullDataset){
//    
//    
//    using namespace std;
//    
//    
//    std::time_t t1 = std::time(0);
//    
//    int frameNumber = 0;
//    // paralelize this loop
//    for (int videoNumber=from; videoNumber<to; videoNumber++) {
//      
//        Struck tracker=Struck::getTracker();
//        tracker.display=0;
//        pair<string, vector<string>> gt_images=video_gt_images[videoNumber];
//        
//        vector<cv::Rect> groundTruth=dataset->readGroundTruth(gt_images.first);
//        
//        frameNumber+=gt_images.second.size();
//        cv::Mat image=cv::imread(gt_images.second[0]);
//        
//        
//        tracker.initialize(image, groundTruth[0]);
//        
//        
//        int nFrames=10;
//        if (fullDataset) {
//            nFrames=gt_images.second.size();
//            
//        }
//        
//        
//        for (int i=1; i<nFrames; i++) {
//            
//            cv::Mat image=cv::imread(gt_images.second[i]);
//            
//            tracker.track(image);
//        }
//        
//        if (saveResults) {
//            std::string saveFileName=saveFolder+"/"+dataset->videos[videoNumber]+".dat";
//            
//            tracker.saveResults(saveFileName);
//        }
//        
//        //tracker.reset();
//        
//        
//    }
//    
//    std::time_t t2 = std::time(0);
//    std::cout<<"Frames per second: "<<frameNumber/(1.0*(t2-t1))<<std::endl;
//    //std::cout<<"No threads: "<<(t2-t1)<<std::endl;
//}