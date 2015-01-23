//
//  DatasetVOT2014.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/22/15.
//
//

#include "DatasetVOT2014.h"
#include <fstream>
#include <regex>
#include <math.h>

//TODO: Implement dataset functions for VOT2014 dataset

std::vector<std::pair<std::string,std::vector<std::string>>> DatasetVOT2014::prepareDataset(std::string rootFolder){
    
    using namespace std;
    
    vector<pair<string, vector<string>>> video_gt_images;
    
    
    // step #1: list all videos
    
    
    vector<string> videos=listSubFolders(rootFolder);
    
    
    for (int i=0; i<(int)videos.size(); i++) {
        
        // create a string which represents ground truth and image location
        string imgLocation=rootFolder+videos[i]+"/";
        
        //TODO: This is not correct
        string gt=rootFolder+videos[i]+"/groundtruth.txt";
        
        // list all images
        
        vector<string> images=Dataset::listImages(imgLocation, "jpg");
        
        std::pair<string, vector<string>> groundTruth_images;
        groundTruth_images=std::make_pair(gt, images);
        
        video_gt_images.push_back(groundTruth_images);
        
    }
    
    
    return video_gt_images;
}

std::vector<cv::Rect> DatasetVOT2014::readGroundTruth(std::string fileName){
    std::vector<cv::Rect> gtRect;
    
    return gtRect;
}

std::vector<cv::RotatedRect> DatasetVOT2014::readComplete(std::string fileName){
    
    
    using namespace std;
    
    
    vector<cv::RotatedRect> gtRect;
    
    std::ifstream infile(fileName);
    string str;
    std::regex e("[[:digit:]]+\\.[[:digit:]]+");
    
    int idx=0;
    
    float num=0;
    while (std::getline(infile, str))
    {
        
        
        std::regex_iterator<std::string::iterator> rit ( str.begin(), str.end(), e );
        std::regex_iterator<std::string::iterator> rend;
        
        int idxValues=0;
        
        std::vector<float> record;
        while (rit!=rend) {
            
            
            num=stof(rit->str());
            
            record.push_back(num);
            ++rit;
        }
        
        // every rectangle is represented as X1, Y1, X2, Y2, X3, Y3, X4, Y4 stored in the vector 'record'
        
        //1) find center -> simply average
        
        float center_x=0;
        float center_y=0;
        
        
        cv::Point2f vertices[4];
        for (int i=0; i<4; i++) {
            center_x+=record[i*2];
            center_y+=record[i*2+1];
            
            vertices[i].x=record[i*2];
            vertices[i].y=record[i*2+1];
        }
        
        center_x/=4;
        center_y/=4;
        
        //2) find width and height ( find for both sides and average)

        
        // (abs(X2-X1)+abs(X4-X3))/2
        float width=sqrtf(powf((vertices[2].x-vertices[1].x),2)+powf((vertices[2].y-vertices[1].y),2));
        float height=sqrtf(powf((vertices[0].x-vertices[2].x),2)+powf((vertices[0].y-vertices[2].y),2));
        
        //3) find the angle
        
        float degrees_1=findAngle(record[0], record[1], record[2], record[3]);
        float degrees_2=findAngle(record[0], record[1], record[2], record[3]);
        
        float angle=degrees_1;
        
        
        
        cv::RotatedRect rot_rect(cv::Point2f(center_x,center_y),cv::Size2f(width,height),angle);
        gtRect.push_back(rot_rect);
  
        ++idx;
    }
    
    
    return gtRect;
    
}


void DatasetVOT2014::showVideo(std::string rootFolder, int vidNumber){
    
    std::vector<std::pair<std::string,std::vector<std::string>>> set=prepareDataset(rootFolder);
    using namespace std;
    using namespace cv;
    
    if (vidNumber<0 || vidNumber>=set.size()) {
        cout<<"Wrong video ID. The ID requested is out of bounds for this dataset."<<endl;
        return;
    }

    vector<cv::RotatedRect> r= readComplete(set[vidNumber].first);
    
    
    vector<string> images=set[vidNumber].second;
    
    
    for (int frame=0; frame<r.size(); frame++) {
        
        cv::Mat M,rotated,cropped;
        RotatedRect rect=r[frame];
        
        float angle=r[frame].angle;
        cv::Size rect_size=r[frame].size;
        
//        if (rect.angle < -45.) {
//            angle += 90.0;
//            swap(rect_size.width, rect_size.height);
//        }
        
        cv::Mat im=cv::imread(images[frame]);
        
        
        
        // get the rotation matrix
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        // perform the affine transformation
        warpAffine(im, rotated, M, im.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated, rect_size, rect.center, cropped);
        
        Point2f vertices[4];
        
        rect.points(vertices);
        for (int i = 0; i < 4; i++){
            line(im, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
            
        }
        
        cv::imshow("cropped", cropped);
        cv::waitKey(2);
        
        cv::imshow("original",im);
        cv::waitKey(3);
    }

}



float DatasetVOT2014::findAngle(float x1,float y1,float x2, float y2){
    
    float deltaY = y2-y1;
    float deltaX = x2-x1;

    
    float angleinDegree=atan2f(deltaX, deltaY)*(180/M_PI);
    
    return -angleinDegree;
    
}



